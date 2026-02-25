#!/usr/bin/env python
##############################################################################
#
# The main computation occurs within a triple nested loop:
# the outer loop iterates over years, the middle loop over days, and
# the innermost loop over grids.
#
# The day and grid loops are moved into the day and grid loops function.
# This is because we use Numba to speed up computations, and
# Numba can only be applied to functions.
#
##############################################################################
from netCDF4 import Dataset
import os
import numpy as np
from datetime import datetime as dt
import warnings
import logging
from numba import njit, prange

from common import (
    Humidity,
    ProcessDailyWind,
    Temperature,
    TemperatureMax,
    TemperatureMin,
    HumidityMax,
    HumidityMin,
    PrecipitationAnnual,
    PrecipitationHours,
)

from NFDRS_functions import(
    MoistureContentMean,
    MoistureContentMax,
    MoistureContentMin,
    DeadOneHourFuelMoisture, 
    DeadTenHourFuelMoisture, 
    DurationDaylight, 
    MoistureContentWeightedAverage, 
    DeadHundredHourFuelMoisture, 
    BoundaryConditionWeightedAverageThousandHour, 
    BoundaryConditionSevenDayAverageThousandHour, 
    DeadThousandHourFuelMoisture, 
    SpreadComponent, 
    EnergyReleaseComponent, 
    BurningIndex, 
    SurfaceWeightedCharacteristicAreaVolumeRatio, 
    HerbaceousIndependentVariablePregreen, 
    MoistureContentHerbaceousPregreen, 
    MoistureContentWoodyPregreen, 
    FrozenPregreen, 
    OneHourFuelLoadingPregreen, 
    HerbaceousFuelLoadingPregreen, 
    HerbaceousIndependentVariableGreenup, 
    MoistureContentHerbaceousGreenup, 
    MoistureContentWoodyGreenup, 
    OneHourFuelLoadingGreenup, 
    HerbaceousFuelLoadingGreenup, 
    HerbaceousIndependentVariableGreen, 
    MoistureContentHerbaceousGreen, 
    MoistureContentWoodyGreen, 
    OneHourFuelLoadingGreen, 
    HerbaceousFuelLoadingGreen, 
    HerbaceousIndependentVariableTransition,
    MoistureContentHerbaceousTransition,
    MoistureContentWoodyTransition,
    FrozenTransition,
    OneHourFuelLoadingTransition,
    HerbaceousFuelLoadingTransition,
    HerbaceousIndependentVariableCured,
    MoistureContentHerbaceousCured,
    MoistureContentWoodyCured,
    FrozenCured,
    OneHourFuelLoadingCured,
    HerbaceousFuelLoadingCured,
    DayAndGridLoops,
    Para,
    Slope,
    Cpara,
)

#########################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s;%(message)s")

sh.setFormatter(formatter)

logger.addHandler(sh)

warnings.filterwarnings("ignore")

#########################################

def create_and_write(varname, arr, std_name):
    var = fileout.createVariable(varname, np.float32, ("time", "y", "x"))
    var.coordinates = "lat lon"
    var.units = ""
    var.standard_name = std_name
    var[:, :, :] = arr

# Start model run
logger.info("Starting the control-control HCLIM3 NFDR model run...")

#########################################

datadir = "/path/to/input/data/directory"

dir_coords = os.path.join(f"{datadir}/path/to/input/climate/data/for/coords.nc")

startyear = 2001
endyear   = 2019

number_of_latitudinal_degrees  = len(Dataset(dir_coords).variables["lat"][:, 0])
number_of_longitudinal_degrees = len(Dataset(dir_coords).variables["lon"][0, :])

number_of_years = endyear - startyear + 1

logger.info("Creating a forest subset...")

dataset_LU = Dataset(
    f"{datadir}/path/to/land_use_file.nc", "r"
)
LU_indices = dataset_LU["Main_Nature_Cover"][:, :]
elevation = dataset_LU["HGT_M"][:, :]
coords_lat = dataset_LU["lat"][:, :]
coords_lon = dataset_LU["lon"][:, :] 

# Numba do not support masked arrays. 
# We will convert these to regulary ndarray. 
if isinstance(LU_indices, np.ma.MaskedArray):
    # Convert to a regular ndarray, replacing masked values with NaN
    LU_indices = LU_indices.filled(np.nan)
    # Now LU_indices is a regular ndarray 

if isinstance(elevation, np.ma.MaskedArray):
    elevation = elevation.filled(np.nan)

if isinstance(coords_lat, np.ma.MaskedArray):
    coords_lat = coords_lat.filled(np.nan)

if isinstance(coords_lon, np.ma.MaskedArray):
    coords_lon = coords_lon.filled(np.nan)

#########################################

all_indexes_lon = []
all_indexes_lat = []
for i in range(0, number_of_latitudinal_degrees - 1):
    for z in range(0, number_of_longitudinal_degrees - 1):
        if (
            LU_indices[i, z] >= 19  # Forests: 19=evergreen, 20=deciduous, 21=mixed
            and LU_indices[i, z] <= 21
        ).all():
            all_indexes_lon.append(z)
            all_indexes_lat.append(i)

latitudes = Dataset(dir_coords).variables["lat"][:, :]
if isinstance(latitudes, np.ma.MaskedArray):
    latitudes = latitudes.filled(np.nan)
  
#########################################

# Setting the length of the grid loop
len_i  = len(all_indexes_lat)

# Setting np arrays used in the grid loop
all_indexes_lon_np = np.array(all_indexes_lon, dtype=int)
all_indexes_lat_np = np.array(all_indexes_lat, dtype=int)

slope_yx = Slope(
    len_i, all_indexes_lon_np, all_indexes_lat_np, elevation,
    number_of_latitudinal_degrees, number_of_longitudinal_degrees)

# Parameters used in the grid loop
# There are 17 parameters:
#
# 0.fuel_model_one_hour_fuel_loading 
# 1.surface_area_volume_one_hour
# 2.fuel_model_ten_hour_fuel_loading 
# 3.surface_area_volume_ten_hour
# 4.fuel_model_hundred_hour_fuel_loading
# 5.surface_area_volume_hundred_hour
# 6.fuel_model_thousand_hour_fuel_loading
# 7.surface_area_volume_thousand_hour
# 8.fuel_model_herbaceous_fuel_loading
# 9.surface_area_volume_herbaceous
# 10.fuel_model_woody_fuel_loading
# 11.surface_area_volume_woody
# 12.fuel_model_drought_fuel_loading
# 13.fuel_bed_depth
# 14.dead_fuel_extinction_moisture
# 15.fuel_heat_combustion
# 16.specified_spread_component

# Parameter values are calculated using numbers.
# It is important to perform these calculations only once, outside the loops.
# All this is done with the function NRIS_para

(para_yx,
 para_LU20_d91,
 para_LU20_else,
 para_LU21_d91,
 para_LU21_else) = Para(
     len_i, all_indexes_lon_np, all_indexes_lat_np, LU_indices,
     number_of_latitudinal_degrees, number_of_longitudinal_degrees)

#########################################


logger.info("Starting the year loop and reading input...")

"""
Assumptions:
- Wind reduction factor is already incorporated in the climate model, 
    and therefore excluded from this model.
- With herbaceous fuel moisture below 30%, in grasslands both heating number 
    herbaceous and woody are 0, and therefore ratio heating numbers cannot be 
    calculated. Assumption is therefore that in this case ratio is calculated 
    by dividing by 0.00056, which is the value the heating number would have with 
    the lowest possible herbaceous fuel moisture.
- Assumptions in the model regarding boundaries for moisture levels in each herbaceous
    phase are unclear. The lowest possible moisture level for wood is therefore set 
    to pregreen levels (80). Only in phase 1 and 5 is herbaceous moisture level allowed
    to sink beneath 30.
- Phase-change is a bit unclear in the model when it comes to switching to phase 2 from 
    1, 3 to 4, or from phase 4 to 5. Assumption is now made that vegetation can only move from 
    phase 1 to phase 2 if frozen_cured is set to 0 (minimum temperatures have been above 
    7.5C for 5 days straight) and moisture levels herbaceous exceed 30. For some locations this
    still leads to no progression to phase 2, which means that an extra condition was set
    where gridcells move up when (arbitrary) day 220 has passed. Vegetation moves to phase 4 
    depending on their latitude and daylight hours. Vegetation needs 3 hours of night time in 
    Northern Norway, 5 in middle Norway and 7 in Southern Norway to go into transition, along 
    with temperatures below 10 degrees to signal oncoming winter. Vegetation moves from phase 4 
    to phase 5 either if frozen_cured is set to 1 (mean temperatures have been below 0C for 5 days 
    straight), when moisture levels sink below 30, or when day 355 has passed. 
    https://www.skogforsk.se/contentassets/ba079a68f2884ef78474594a03bfae65/plantskolan-no-17-short-day-treatment-final.pdf
- Due to missing information on slopes per gridcell, elevation levels were extrapolated from 
    the HiddenCosts WRF data. Slopes were then calculated by taking the maximum difference in 
    elevation between the gridcell in question and neighbouring gridcells, dividing this by 
    the distance between gridcells (3km) and taking the arctan to convert to degrees.
- Climate class is determined based on annual precipitation for each gridcell.
"""

years = range(startyear, endyear)
for year in years:
    logger.info(f"Starting loop for year {year} ---") 
    if year % 4 == 0:
        days_in_year = 366
    else:
        days_in_year = 365

    temperature_mean_input = TemperatureMean(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/tas_daymean_{year}_Fennoscandia.nc"),
    )
	temperature_mean_input = temperature_mean_input * (9 / 5) + 32 #Conversion Celsius to Fahrenheit
    #
    # Numba do not support masked arrays.
    # We will convert these to regulary ndarray.
    #
    if isinstance(temperature_mean_input, np.ma.MaskedArray):
        # Convert to a regular ndarray, replacing masked values with NaN
        temperature_mean_input = temperature_mean_input.filled(np.nan)
        # Now temperature_mean_input is a regular ndarray
    
    temperature_max_input = TemperatureMax(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/tas_daymax_{year}_Fennoscandia.nc"),
    )
    if isinstance(temperature_max_input, np.ma.MaskedArray):  
        temperature_max_input = temperature_max_input.filled(np.nan)
      
    temperature_min_input = TemperatureMin(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/tas_daymin_{year}_Fennoscandia.nc"),
    )
    if isinstance(temperature_min_input, np.ma.MaskedArray):
        temperature_min_input = temperature_min_input.filled(np.nan)
      
    relative_humidity_at_2m_input = Humidity(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/hurs_daymean_{year}_Fennoscandia.nc"),
    )
    if isinstance(relative_humidity_at_2m_input, np.ma.MaskedArray):
        relative_humidity_at_2m_input = relative_humidity_at_2m_input.filled(np.nan)
      
    humidity_min_input = HumidityMin(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/hurs_daymin_{year}_Fennoscandia.nc"),
    )
    if isinstance(humidity_min_input, np.ma.MaskedArray):
        humidity_min_input = humidity_min_input.filled(np.nan)
      
    humidity_max_input = HumidityMax(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/hurs_daymax_{year}_Fennoscandia.nc"),
    )
    if isinstance(humidity_max_input, np.ma.MaskedArray):
        humidity_max_input = humidity_max_input.filled(np.nan)

    (wind_speeds_at_10m_input) = (
        ProcessDailyWind(
            dir_u10=os.path.join(
                f"{datadir}/input/HCLIM3output/{year}/uas_hourly_{year}_Fennoscandia.nc"
            ),
            dir_v10=os.path.join(
                f"{datadir}/input/HCLIM3output/{year}/vas_hourly_{year}_Fennoscandia.nc"
            ),
        )
    ) 
    if isinstance(wind_speeds_at_10m_input, np.ma.MaskedArray):
        wind_speeds_at_10m_input = wind_speeds_at_10m_input.filled(np.nan)
      
    precipitation_input = PrecipitationHours(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/pr_duration_{year}_Fennoscandia.nc"),
    )
    if isinstance(precipitation_input, np.ma.MaskedArray):
        precipitation_input = precipitation_input.filled(np.nan)
    
    precipitation_annual_input = PrecipitationAnnual(
        os.path.join(f"{datadir}/input/HCLIM3output/{year}/pr_yearsum_{year}_Fennoscandia.nc"),
        HCLIM3=True,
    )
    if isinstance(precipitation_annual_input, np.ma.MaskedArray):
        precipitation_annual_input = precipitation_annual_input.filled(np.nan)

    #########################################
  
    duration_daylight_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    climate_class_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    index_y_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    index_x_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    day_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    phase_herbaceous_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    frozen_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    day_change_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    dead_one_hour_fuel_moisture_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    dead_ten_hour_fuel_moisture_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    dead_hundred_hour_fuel_moisture_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    dead_thousand_hour_fuel_moisture_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    boundary_condition_weighted_average_thousand_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    herbaceous_independent_variable_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    boundary_condition_seven_day_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    moisture_content_herbaceous_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    moisture_content_woody_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    spread_component_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    energy_release_component_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    burning_index_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    burning_category_output = np.full(
        (days_in_year, number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    #
    # The function DayAndGrid_loops uses the arrays:
    # dead_hundred_hour_fuel_moisture_input
    # dead_thousand_hour_fuel_moisture_input
    #
    # These are used in an if test starting with:
    # elif year > startyear and day == 0:
    #
    # The function DayAndGrid_loops also uses the array:
    # boundary_condition_seven_day_input
    #
    # This array is used in an if test starting with:
    # if year - 1 % 4 == 0:
    #
    # These arrays must be passed to the DayAndGrid_loops function.
    # If they are not created below in the if test (if year>2001),
    # we need to ensure they exist by initializing them as empty arrays.
    # These empty arrays can then be filled with values in the if test below.
    #
    dead_hundred_hour_fuel_moisture_input = np.full(
        (number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    dead_thousand_hour_fuel_moisture_input = np.full(
	  (number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )
    boundary_condition_seven_day_input = np.full(
        (number_of_latitudinal_degrees, number_of_longitudinal_degrees), np.nan
    )

    #########################################
  
    if year > startyear:
        prev_year_path = f"{datadir}/path/to/previous/year/outfile_{year-1}.nc"
        if os.path.exists(prev_year_path):
            file_in = Dataset(prev_year_path)
            boundary_condition_weighted_average_thousand_input = file_in[
                "boundary_condition_weighted_average_thousand"
            ][:, :, :]
            if isinstance(boundary_condition_weighted_average_thousand_input, np.ma.MaskedArray):
                boundary_condition_weighted_average_thousand_input = boundary_condition_weighted_average_thousand_input.filled(1.e+20)
            
            if year - 1 % 4 == 0:
                dead_hundred_hour_fuel_moisture_input = file_in[
                    "dead_hundred_hour_fuel_moisture"
                ][365, :, :]
                dead_thousand_hour_fuel_moisture_input = file_in[
                    "dead_thousand_hour_fuel_moisture"
                ][365, :, :]
                boundary_condition_seven_day_input = file_in[
                    "boundary_condition_weighted_average_thousand"
                ][365, :, :]
            else:
                dead_hundred_hour_fuel_moisture_input = file_in[
                    "dead_hundred_hour_fuel_moisture"
                ][364, :, :]
                dead_thousand_hour_fuel_moisture_input = file_in[
                    "dead_thousand_hour_fuel_moisture"
                ][364, :, :]
                boundary_condition_seven_day_input = file_in[
                    "boundary_condition_weighted_average_thousand"
                ][364, :, :]

            if isinstance(dead_hundred_hour_fuel_moisture_input, np.ma.MaskedArray):
                dead_hundred_hour_fuel_moisture_input = dead_hundred_hour_fuel_moisture_input.filled(np.nan)
            if isinstance(dead_thousand_hour_fuel_moisture_input, np.ma.MaskedArray):
                dead_thousand_hour_fuel_moisture_input = dead_thousand_hour_fuel_moisture_input.filled(np.nan)
            if isinstance(boundary_condition_seven_day_input, np.ma.MaskedArray):
                boundary_condition_seven_day_input = boundary_condition_seven_day_input.filled(np.nan)   # or np.nan, or whatever makes sense
              
        else:
            logger.info(f"CRITICAL: Resource for {year-1} not found at {prev_year_path}")

    #########################################
  
    phase_herbaceous = 1
    moisture_content_herbaceous = 30
    day_change = 0
    #
    # There are 13 Climate class parameters:
    #
    # 0.climate_class
    # 1.herbaceous_greenup_constant
    # 2.herbaceous_greenup_coefficient
    # 3.annual_herbaceous_transition_constant
    # 4.annual_herbaceous_transition_coefficient
    # 5.perennial_herbaceous_transition_constant
    # 6.perennial_herbaceous_transition_coefficient
    # 7.moisture_content_wood_pregreen_stage
    # 8.wood_greenup_constant
    # 9.wood_greenup_coefficient
    # 10.rainfall_rate
    # 11.greenup_period_weeks
    # 12.greenup_period_days
    #
    # This removes many if-test from the grid loop.
    # These parameters are computed by the NRIS_Cpara function.

    Cpara_yx = Cpara(
        len_i, all_indexes_lon_np, all_indexes_lat_np,
        number_of_latitudinal_degrees, number_of_longitudinal_degrees,
        precipitation_annual_input)  
  
    logger.info(f"Starting the day and grid loops for year {year}...")
  
    #########################################
  
    (
    duration_daylight_output,
    climate_class_output,
    index_y_output,
    index_x_output,
    day_output,
    day_change_output,
    frozen_output,
    phase_herbaceous_output,
    dead_one_hour_fuel_moisture_output,
    dead_ten_hour_fuel_moisture_output,
    dead_hundred_hour_fuel_moisture_output,
    dead_thousand_hour_fuel_moisture_output,
    boundary_condition_weighted_average_thousand_output,
    boundary_condition_seven_day_output,
    herbaceous_independent_variable_output,
    moisture_content_herbaceous_output,
    moisture_content_woody_output,
    spread_component_output,
    energy_release_component_output,
    burning_index_output,
    burning_category_output
    ) = DayAndGridLoops(
        days_in_year, len_i, all_indexes_lon_np, all_indexes_lat_np, latitudes, slope_yx, para_yx, LU_indices,
        para_LU20_d91, para_LU20_else, para_LU21_d91, para_LU21_else, temperature_mean_input,
        relative_humidity_at_2m_input, humidity_min_input, humidity_max_input, wind_speeds_at_10m_input,
        precipitation_input, temperature_max_input, temperature_min_input, Cpara_yx, year, startyear,
        dead_hundred_hour_fuel_moisture_output, dead_thousand_hour_fuel_moisture_output,
        boundary_condition_weighted_average_thousand_output, boundary_condition_seven_day_output,
        phase_herbaceous_output, frozen_output, day_change_output,
        herbaceous_independent_variable_output, moisture_content_herbaceous_output,
        moisture_content_woody_output, spread_component_output, energy_release_component_output,
        burning_index_output, burning_category_output, duration_daylight_output, climate_class_output,
        index_y_output, index_x_output, day_output, dead_one_hour_fuel_moisture_output,
        dead_ten_hour_fuel_moisture_output,
        dead_hundred_hour_fuel_moisture_input, dead_thousand_hour_fuel_moisture_input,
        boundary_condition_seven_day_input)
    
    #########################################
    logger.info("Starting to write output...")
    
    filepath = os.path.join(
        f"{datadir}/path/to/outfile_{year}.nc"
    )

    toinclude = ["lon", "lat", "time_bnds", "time"]

    with Dataset(
        os.path.join(f"{datadir}/path/to/template/hurs_daymean_{year}.nc"), "r"
    ) as filein, Dataset(filepath, "w", format="NETCDF4") as fileout:
        for name in filein.ncattrs():
            fileout.setncattr(name, filein.getncattr(name))
        for name, dimension in filein.dimensions.items():
            if dimension.isunlimited():
                fileout.createDimension(name, None)
            else:
                fileout.createDimension(name, len(dimension))
        for name, variable in filein.variables.items():
            if name in toinclude:
                x = fileout.createVariable(name, variable.datatype, variable.dimensions)
                x.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                x[:] = variable[:]

        create_and_write("duration_daylight", duration_daylight_output, "duration_daylight")
        create_and_write("climate_class", climate_class_output, "climate_class")
        create_and_write("index_y",index_y_output,"index_y")
        create_and_write("index_x",index_x_output,"index_x")
        create_and_write("day",day_output,"day")
        create_and_write("day_change",day_change_output,"day_change")
        create_and_write("frozen_category",frozen_output,"frozen_category")
        create_and_write("phase_herbaceous",phase_herbaceous_output,"phase_herbaceous")
        create_and_write("dead_one_hour_fuel_moisture",dead_one_hour_fuel_moisture_output,"dead_one_hour_fuel_moisture")
        create_and_write("dead_ten_hour_fuel_moisture",dead_ten_hour_fuel_moisture_output,"dead_ten_hour_fuel_moisture")
        create_and_write("dead_hundred_hour_fuel_moisture",dead_hundred_hour_fuel_moisture_output,"dead_hundred_hour_fuel_moisture")
        create_and_write("dead_thousand_hour_fuel_moisture",dead_thousand_hour_fuel_moisture_output,"dead_thousand_hour_fuel_moisture")
        create_and_write("boundary_condition_weighted_average_thousand",boundary_condition_weighted_average_thousand_output,"boundary_condition_weighted_average_thousand")
        create_and_write("boundary_condition_seven_day",boundary_condition_seven_day_output,"boundary_condition_seven_day")
        create_and_write("herbaceous_independent_variable",herbaceous_independent_variable_output,"herbaceous_independent_variable")
        create_and_write("moisture_content_herbaceous",moisture_content_herbaceous_output,"moisture_content_herbaceous")
        create_and_write("moisture_content_woody",moisture_content_woody_output,"moisture_content_woody")
        create_and_write("spread_component",spread_component_output,"spread_component")
        create_and_write("energy_release_component",energy_release_component_output,"energy_release_component")
        create_and_write("burning_index",burning_index_output,"burning_index")
        create_and_write("burning_category",burning_category_output,"burning_category")
        
    logger.info(f"{year} done!")

