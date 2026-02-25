#!/usr/bin/env python3
import logging
import os
import warnings
from math import exp, log, sin, acos, tan, radians
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime
import pandas as pd
from numba import njit, prange
from common import (
    Humidity,
    Precipitation,
    Temperature,
    ProcessDailyWind,
)
warnings.filterwarnings("ignore")

# -------------------------
# Configure logging
# -------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)

# -------------------------
# FWI core functions (refactored, variable names kept)
# -------------------------
@njit(fastmath=True)
def FineFuelMoistureCode(
    starting_fine_fuel_moisture_code, precipitation, humidities, temperature, windspeeds
):
    starting_fine_fuel_moisture_code_denominator = 59.5 + starting_fine_fuel_moisture_code
    if starting_fine_fuel_moisture_code_denominator < 1e-6:
        starting_fine_fuel_moisture_code_denominator = 1e-6    
    starting_moisture_content = (
        147.2
        * (101.0 - starting_fine_fuel_moisture_code)
        / starting_fine_fuel_moisture_code_denominator
    )  # Eq. 1

    if starting_moisture_content > 250.0:
        starting_moisture_content = 250.0
    elif starting_moisture_content < 0.0:
        starting_moisture_content = 0.0
    elif not np.isfinite(starting_moisture_content):
        starting_moisture_content = 0.0

    # Rain effect
    if precipitation > 0.5:  # Eq. 2,3
        starting_moisture_content_denominator = (251.0 - starting_moisture_content)
        if starting_moisture_content_denominator < 1e-6:
            starting_moisture_content_denominator = 1e-6
        rain = max(precipitation - 0.5, 1e-6)
        if starting_moisture_content > 150.0:
            starting_moisture_content = (
                starting_moisture_content
                + 42.5
                * rain
                * exp(-100.0 / starting_moisture_content_denominator)
                * (1.0 - exp(-6.93 / rain))
            ) + 0.0015 * ((starting_moisture_content - 150.0) ** 2) * (rain**0.5)
        else:
            starting_moisture_content = starting_moisture_content + 42.5 * rain * exp(
                -100.0 / starting_moisture_content_denominator
            ) * (1.0 - exp(-6.93 / rain))

    if starting_moisture_content > 250.0:
        starting_moisture_content = 250.0
    elif starting_moisture_content < 0.0:
        starting_moisture_content = 0.0
    elif not np.isfinite(starting_moisture_content):
        starting_moisture_content = 0.0

    equilibrium_moisture_content_drying = (
        0.942 * (humidities**0.679)
        + 11.0 * exp((humidities - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature) * (1.0 - exp(-0.115 * humidities))
    )  # Eq. 4

    equilibrium_moisture_content_wetting = (
        0.618 * (humidities**0.753)
        + 10.0 * exp((humidities - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature) * (1.0 - exp(-0.115 * humidities))
    )  # Eq. 5

    if starting_moisture_content > equilibrium_moisture_content_drying:
        log_drying_rate = (
            (
                0.424 * (1.0 - (humidities / 100.0) ** 1.7)
                + (0.0694 * (windspeeds**0.5)) * (1.0 - (humidities / 100.0) ** 8)
            )
            * 0.581
            * exp(0.0365 * temperature)
        )  # Eq. 6
        moisture_content = equilibrium_moisture_content_drying + (
            starting_moisture_content - equilibrium_moisture_content_drying
        ) * 10.0 ** (-log_drying_rate)  # Eq. 8
    elif starting_moisture_content < equilibrium_moisture_content_drying:
        if starting_moisture_content < equilibrium_moisture_content_wetting:
            log_wetting_rate = (
                0.424 * (1.0 - ((100.0 - humidities) / 100.0) ** 1.7)
                + (0.0694 * (windspeeds**0.5))
                * (1.0 - ((100.0 - humidities) / 100.0) ** 8)
            ) * (0.581 * exp(0.0365 * temperature))  # Eq. 7
            moisture_content = equilibrium_moisture_content_wetting - (
                equilibrium_moisture_content_wetting - starting_moisture_content
            ) * 10.0 ** (-log_wetting_rate)  # Eq. 9
        else:
            moisture_content = starting_moisture_content
    else:
        moisture_content = starting_moisture_content

    if moisture_content < 0:
        moisture_content = 0.0
    elif not np.isfinite(moisture_content):
        moisture_content = 0.0

    todays_fine_fuel_moisture_code_denominator = 147.2 + moisture_content
    if todays_fine_fuel_moisture_code_denominator < 1e-6:
        todays_fine_fuel_moisture_code_denominator = 1e-6
    todays_fine_fuel_moisture_code = (59.5 * (250.0 - moisture_content)) / todays_fine_fuel_moisture_code_denominator # Eq. 10

    if todays_fine_fuel_moisture_code > 101.0:
        todays_fine_fuel_moisture_code = 101.0
    elif todays_fine_fuel_moisture_code < 0.0:
        todays_fine_fuel_moisture_code = 0.0

    return todays_fine_fuel_moisture_code

@njit(fastmath=True)
def DurationDaylight(latitude, day_of_year):
    """
    duration_daylight expects day_of_year (1..365/366).
    """
    phi = radians(latitude)
    solar_declination = 0.41008 * sin(radians((day_of_year - 82.0)))
    x = tan(phi) * tan(solar_declination)

    if x < -1.0:
        duration_daylight = 0.0
    elif x > 1.0:
        duration_daylight = 24.0
    else:
        duration_daylight = 24.0 * (1.0 - (acos(x) / 3.1416))
    return duration_daylight

@njit(fastmath=True)
def DuffMoistureCode(
    temperature,
    precipitation,
    starting_duff_moisture_code,
    humidities,
    duration_daylight,
):
    if temperature < -1.1:
        temperature = -1.1

    if precipitation > 1.5:
        rain = precipitation
        effective_rain = 0.92 * rain - 1.27  # Eq. 11

        # Eq. 12 alternate form used in your earlier refactor:
        initial_moisture_index = 20.0 + 280.0 * exp(-0.023 * starting_duff_moisture_code)
        
        if starting_duff_moisture_code <= 33.0:
            function_rain_effect_denominator = 0.5 + (0.3 * starting_duff_moisture_code)
            if function_rain_effect_denominator < 1e-6:
                function_rain_effect_denominator = 1e-6                         
            function_rain_effect = 100.0 / function_rain_effect_denominator  # Eq. 13a
        elif starting_duff_moisture_code <= 65.0:
            function_rain_effect = 14.0 - (1.3 * log(starting_duff_moisture_code))  # Eq. 13b
        else:
            function_rain_effect = (6.2 * log(starting_duff_moisture_code)) - 17.2  # Eq. 13c

        moisture_content_after_rain_denominator = (48.77 + function_rain_effect * effective_rain)
        if moisture_content_after_rain_denominator < 1e-6:
            moisture_content_after_rain_denominator = 1e-6
        moisture_content_after_rain = initial_moisture_index + (
            (1000.0 * effective_rain) / moisture_content_after_rain_denominator
        )  # Eq. 14

        starting_duff_moisture_code = 43.43 * (5.6348 - log(moisture_content_after_rain - 20.0))  # Eq. 15

    if starting_duff_moisture_code < 0.0:
        starting_duff_moisture_code = 0.0

    # Eq. 16 drying factor (note: your previous code used 0.0001 scale)
    drying_factor = (
        1.894
        * (temperature + 1.1)
        * (100.0 - humidities)
        * (duration_daylight * 0.0001)
    )
    duff_moisture_code = starting_duff_moisture_code + drying_factor

    if duff_moisture_code <= 1.0:
        duff_moisture_code = 1.0
    return duff_moisture_code

@njit(fastmath=True)
def DroughtCode(year, day_of_year, temperature, precipitation, starting_drought_code):
    """
    Follows Eq. 18-22 from Van Wagner.
    day_of_year is 1..365/366.
    """
    # Determine month index
    if year % 4 == 0:
        month_limits = [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    else:
        month_limits = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    # find month index 0..11
    month = 0
    for i in range(12):
        if day_of_year <= month_limits[i]:
            month = i
            break
        # next(i for i, lim in enumerate(month_limits) if day_of_year <= lim)

    daylength_adjustment = [
        -1.6,
        -1.6,
        -1.6,
        0.9,
        3.8,
        5.8,
        6.4,
        5.0,
        2.4,
        0.4,
        -1.6,
        -1.6,
    ]

    if temperature < -2.8:
        temperature = -2.8

    potential_evapotranspiration = (
        0.36 * (temperature + 2.8) + daylength_adjustment[month]
    ) / 2.0
    if potential_evapotranspiration < 0.0:
        potential_evapotranspiration = 0.0

    if precipitation > 2.8:
        effective_rain = 0.83 * precipitation - 1.27  # Eq. 18
        moisture_equivalent_drought_code = 800.0 * exp(-starting_drought_code / 400.0)  # Eq. 19
        moisture_equivalent_drought_code_after_rain = moisture_equivalent_drought_code + 3.937 * effective_rain  # Eq. 20
        drought_code = 400.0 * log(800.0 / moisture_equivalent_drought_code_after_rain)  # Eq. 21
        if drought_code < 0.0:
            drought_code = 0.0

    drought_code = drought_code + potential_evapotranspiration  # Eq. 22
    return drought_code

@njit(fastmath=True)
def InitialSpreadIndex(todays_fine_fuel_moisture_code, windspeeds):
    wind_function = exp(0.05039 * windspeeds)

    moisture_denominator = (59.5 + todays_fine_fuel_moisture_code)
    if moisture_denominator < 1e-6:
        moisture_denominator = 1e-6
    moisture = (
        147.2
        * (101.0 - todays_fine_fuel_moisture_code)
        / moisture_denominator
    )
    fine_fuel_moisture_function = 91.9 * exp(-0.1386 * moisture) * (1.0 + (moisture**5.31) / 4.93e7)
    initial_spread_index = 0.208 * wind_function * fine_fuel_moisture_function
    return initial_spread_index

@njit(fastmath=True)
def BuildupIndex(duff_moisture_code, drought_code):
    buildup_index_denominator = (duff_moisture_code + 0.4 * drought_code)
    if buildup_index_denominator < 1e-6:
        buildup_index_denominator = 1e-6

    if duff_moisture_code == 0.0 and drought_code == 0.0:
        return 0.0
    if duff_moisture_code > 0.4 * drought_code:
        buildup_index = duff_moisture_code - (
            1.0 - 0.8 * drought_code / buildup_index_denominator
        ) * (0.92 + (0.0114 * duff_moisture_code) ** 1.7)
    else:
        buildup_index = (0.8 * drought_code * duff_moisture_code) / buildup_index_denominator
    if buildup_index < 0.0:
        buildup_index = 0.0
    return buildup_index

@njit(fastmath=True)
def FireWeatherIndex(buildup_index, initial_spread_index):
    if buildup_index > 80.0:
        duff_moisture_function_denominator = (25.0 + 108.64 * exp(-0.023 * buildup_index))
        if duff_moisture_function_denominator < 1e-6:
            duff_moisture_function_denominator = 1e-6
        duff_moisture_function = 1000.0 / duff_moisture_function_denominator
    else:
        duff_moisture_function = 0.626 * (buildup_index**0.809) + 2.0
    intermediate_fire_weather_index = 0.1 * initial_spread_index * duff_moisture_function
    if intermediate_fire_weather_index <= 1.0:
        return intermediate_fire_weather_index
    else:
        return exp(2.72 * (0.434 * log(intermediate_fire_weather_index)) ** 0.647)

@njit(fastmath=True)
def FireHazardRating(fwi):
    """
    Simple categorical hazard mapping.
    """
    if fwi == 0:
        return 0
    elif fwi <= 5:
        return 1
    elif fwi <= 10:
        return 2
    elif fwi <= 20:
        return 3
    elif fwi <= 30:
        return 4
    else:
        return 5


# -------------------------
# Main driver (WRF/HCLIM compatible)
# -------------------------
def run_fwi_hclim(
    datadir,
    dir_coords,
    startyear,
    endyear,
    forest_classes=(
        19,
        20,
        21,
    ),
):
    """
    datadir: base directory with input and output subfolders
    dir_coords: path to the coords and landuse netcdf (Main_Nature_Cover, lat, lon)
    startyear, endyear: inclusive start, exclusive end
    forest_classes: land-cover classes to include (coniferous forests, deciduous forests, mixed forests)
    """
    coords_dataset = Dataset(dir_coords)
    LU_indices = coords_dataset["Main_Nature_Cover"][:, :]
    lats = coords_dataset.variables["lat"][:, :]
    # lons = coords_dataset.variables["lon"][:, :]

    # Build index list for desired landcover (fixed selection logic)
    all_indexes_lon = []
    all_indexes_lat = []
    ny, nx = LU_indices.shape
    for iy in range(ny):
        for ix in range(nx):
            if LU_indices[iy, ix] in forest_classes:
                all_indexes_lon.append(ix)
                all_indexes_lat.append(iy)
    logger.info(f"FWI: Found {len(all_indexes_lon)} grid points in selected classes")

    years = range(startyear, endyear)
    for year in years:
        logger.info(f"FWI: Starting year {year}")
        # days in year
        days_in_year = 366 if (year % 4 == 0) else 365

        # arrays: time, y, x. Use float32 and NaN default.
        shape = (days_in_year, ny, nx)
        initial_spread_index_output = np.full(shape, np.nan, dtype=np.float32)
        buildup_index_output = np.full(shape, np.nan, dtype=np.float32)
        fire_weather_index_output = np.full(shape, np.nan, dtype=np.float32)
        fire_hazard_rating_output = np.full(shape, np.nan, dtype=np.float32)
        starting_fine_fuel_moisture_code_output = np.full(
            shape, np.nan, dtype=np.float32
        )
        starting_duff_moisture_code_output = np.full(shape, np.nan, dtype=np.float32)
        starting_drought_code_output = np.full(shape, np.nan, dtype=np.float32)
        daily_severity_rating_output = np.full(shape, np.nan, dtype=np.float32)

        # initial arrays for the grid
        starting_fine_fuel_moisture_code_input = np.full(
            (ny, nx), 85.0, dtype=np.float32
        )
        starting_duff_moisture_code_input = np.full((ny, nx), 6.0, dtype=np.float32)
        starting_drought_code_input = np.full((ny, nx), 15.0, dtype=np.float32)

        # Read climate inputs
        temperature_at_2m_input = Temperature(
            os.path.join(f"{datadir}/path/to/tas_daymean_{year}.nc"),
        )
        relative_humidity_at_2m_input = Humidity(
            os.path.join(f"{datadir}/path/to/hurs_daymean_{year}.nc"),
        )
        (wind_speeds_at_10m_input) = (
            ProcessDailyWind(
                dir_u10=os.path.join(
                    f"{datadir}/path/to/uas_hourly_{year}.nc"
                ),
                dir_v10=os.path.join(
                    f"{datadir}/path/to/vas_hourly_{year}.nc"
                ),
            )
        )
        precipitation_input = Precipitation(
            os.path.join(f"{datadir}/path/to/pr_daysum_{year}.nc"),
        )

        # If not the first year, try to read previous year's final values for initialization
        if year > 2001:
            prev_path = (
                f"{datadir}/path/to/output/file/previous/year/outfile_{year-1}.nc"
            )
            if os.path.exists(prev_path):
                with Dataset(prev_path, "r") as file_in:
                    prev_days = 366 if ((year - 1) % 4 == 0) else 365
                    last_index = prev_days - 1
                    # defensive: ensure vars exist
                    if "starting_fine_fuel_moisture_code" in file_in.variables:
                        starting_fine_fuel_moisture_code_input[:, :] = file_in[
                            "starting_fine_fuel_moisture_code"
                        ][last_index, :, :]
                    if "starting_duff_moisture_code" in file_in.variables:
                        starting_duff_moisture_code_input[:, :] = file_in[
                            "starting_duff_moisture_code"
                        ][last_index, :, :]
                    if "starting_drought_code" in file_in.variables:
                        starting_drought_code_input[:, :] = file_in[
                            "starting_drought_code"
                        ][last_index, :, :]

                    if isinstance(starting_fine_fuel_moisture_code_input, np.ma.MaskedArray):
                        starting_fine_fuel_moisture_code_input = starting_fine_fuel_moisture_code_input.filled(1.e+20)
                    if isinstance(starting_duff_moisture_code_input, np.ma.MaskedArray):
                        starting_duff_moisture_code_input = starting_duff_moisture_code_input.filled(1.e+20)
                    if isinstance(starting_drought_code_input, np.ma.MaskedArray):
                        starting_drought_code_input = starting_drought_code_input.filled(1.e+20)
            
            else:
                logger.info("FWI: Previous-year file not found; using defaults")

        # Loop over days and selected grid points
        for pt in prange(len(all_indexes_lon)):
            for day_index in range(days_in_year):
                # convert 0-based day_index to 1-based day_of_year (DurationDaylight and DroughtCode expect 1..365/366)
                day_of_year = day_index + 1
            
                ix = all_indexes_lon[pt]
                iy = all_indexes_lat[pt]

                latitude = float(lats[iy, ix])

                starting_fine_fuel_moisture_code = float(
                    starting_fine_fuel_moisture_code_input[iy, ix]
                )
                if starting_fine_fuel_moisture_code > 101.0:
                    starting_fine_fuel_moisture_code = 101.0
                elif starting_fine_fuel_moisture_code < 0.0:
                    starting_fine_fuel_moisture_code = 0.0
                elif not np.isfinite(starting_fine_fuel_moisture_code):
                    starting_fine_fuel_moisture_code = 0.0
                    
                starting_duff_moisture_code = float(
                    starting_duff_moisture_code_input[iy, ix]
                )
                starting_drought_code = float(starting_drought_code_input[iy, ix])

                # Read model fields at (time, y, x)
                temperature = float(temperature_at_2m_input[day_index, iy, ix])
                humidities = float(relative_humidity_at_2m_input[day_index, iy, ix])
                windspeeds = (
                    float(wind_speeds_at_10m_input[day_index, iy, ix]) * 3.6
                )  # m/s to km/h
                precipitation = float(precipitation_input[day_index, iy, ix])

                # Sanity clamp RH
                if humidities > 100.0:
                    humidities = 100.0
                elif humidities < 0.0:
                    humidities = 0.0

                # FFMC (updates starting_fine_fuel_moisture_code)
                todays_fine_fuel_moisture_code = FineFuelMoistureCode(
                    starting_fine_fuel_moisture_code,
                    precipitation,
                    humidities,
                    temperature,
                    windspeeds,
                )

                # Duration daylight (for DMC)
                duration_daylight = DurationDaylight(latitude, day_of_year)

                # DMC
                duff_moisture_code = DuffMoistureCode(
                    temperature,
                    precipitation,
                    starting_duff_moisture_code,
                    humidities,
                    duration_daylight,
                )

                # DC (note: we pass day_of_year and year)
                drought_code = DroughtCode(
                    year, day_of_year, temperature, precipitation, starting_drought_code
                )

                # ISI, BUI, FWI
                initial_spread_index = InitialSpreadIndex(
                    todays_fine_fuel_moisture_code, windspeeds
                )
                buildup_index = BuildupIndex(duff_moisture_code, drought_code)
                fire_weather_index = FireWeatherIndex(
                    buildup_index, initial_spread_index
                )
                fire_hazard_rating = FireHazardRating(fire_weather_index)
                daily_severity_rating = 0.0272 * (fire_weather_index**1.77)

                # Store outputs
                initial_spread_index_output[day_index, iy, ix] = initial_spread_index
                buildup_index_output[day_index, iy, ix] = buildup_index
                fire_weather_index_output[day_index, iy, ix] = fire_weather_index
                fire_hazard_rating_output[day_index, iy, ix] = fire_hazard_rating
                daily_severity_rating_output[day_index, iy, ix] = daily_severity_rating

                # Update starting codes for next day at this gridpoint
                starting_fine_fuel_moisture_code = todays_fine_fuel_moisture_code
                starting_duff_moisture_code = duff_moisture_code
                starting_drought_code = drought_code

                starting_fine_fuel_moisture_code_input[iy, ix] = (
                    starting_fine_fuel_moisture_code
                )
                starting_duff_moisture_code_input[iy, ix] = starting_duff_moisture_code
                starting_drought_code_input[iy, ix] = starting_drought_code

                starting_fine_fuel_moisture_code_output[day_index, iy, ix] = (
                    starting_fine_fuel_moisture_code
                )
                starting_duff_moisture_code_output[day_index, iy, ix] = (
                    starting_duff_moisture_code
                )
                starting_drought_code_output[day_index, iy, ix] = starting_drought_code

            # end gridpoint loop
        # end day loop

        # Write outputs to netCDF (use one of the model input files as a template for dims/attrs)
        filepath_out = os.path.join(
            f"{datadir}/path/to/output/outfile_{year}.nc"
        )
        template_in = os.path.join(
            f"{datadir}/path/to/input/template/infile_{year}.nc"
        )
        toinclude = ["lon", "lat", "time_bnds", "time"]
        with Dataset(template_in, "r") as filein, Dataset(
            filepath_out, "w", format="NETCDF4"
        ) as fileout:
            # copy global attributes
            for name in filein.ncattrs():
                fileout.setncattr(name, filein.getncattr(name))
            # copy dimensions
            for name, dimension in filein.dimensions.items():
                if dimension.isunlimited():
                    fileout.createDimension(name, None)
                else:
                    fileout.createDimension(name, len(dimension))
            # copy selected variables
            for name, variable in filein.variables.items():
                if name in toinclude:
                    x = fileout.createVariable(
                        name, variable.datatype, variable.dimensions
                    )
                    x.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                    x[:] = variable[:]

            def create_and_write(varname, arr, std_name):
                var = fileout.createVariable(varname, np.float32, ("time", "y", "x"), fill_value=1.e+20)
                var.coordinates = "lat lon"
                var.units = ""
                var.standard_name = std_name
                var[:, :, :] = arr

            create_and_write(
                "initial_spread_index",
                initial_spread_index_output,
                "initial_spread_index",
            )
            create_and_write("buildup_index", buildup_index_output, "buildup_index")
            create_and_write(
                "daily_severity_rating",
                daily_severity_rating_output,
                "daily_severity_rating",
            )
            create_and_write(
                "starting_fine_fuel_moisture_code",
                starting_fine_fuel_moisture_code_output,
                "starting_fine_fuel_moisture_code",
            )
            create_and_write(
                "starting_duff_moisture_code",
                starting_duff_moisture_code_output,
                "starting_duff_moisture_code",
            )
            create_and_write(
                "starting_drought_code",
                starting_drought_code_output,
                "starting_drought_code",
            )
            create_and_write(
                "fire_weather_index", fire_weather_index_output, "fire_weather_index"
            )
            create_and_write(
                "fire_hazard_rating", fire_hazard_rating_output, "fire_hazard_rating"
            )

        logger.info(f"FWI: Wrote outputs to {filepath}")

    logger.info("FWI: All years processed.")


# -------------------------
# Run if executed as script
# -------------------------
if __name__ == "__main__":
    logger.info("FWI: Starting the HCLIM3 FWI model run...")
    datadir = "/path/to/climate/data"
    dir_coords = os.path.join(
        f"{datadir}/land_use_file.nc"
    )

    # Example years - adjust to your original values
    run_fwi_hclim(datadir, dir_coords, startyear=2001, endyear=2019)
    logger.info("FWI: Done!")
