import numpy as np
from netCDF4 import Dataset, num2date
import math
import pandas as pd
from datetime import datetime
from numba import njit, prange

# -------------------------
# FWI climate input functions
# -------------------------
def Precipitation(dir_precipitation):
    with Dataset(dir_precipitation) as file_precipitation:
        return file_precipitation.variables["pr"][:, :, :]

def Temperature(dir_temperature):
    with Dataset(dir_temperature) as file_temperature_at_2m:
        return file_temperature_at_2m.variables["tas"][:, :, :] - 273.15

def Humidity(dir_humidity):
    with Dataset(dir_humidity) as file_relative_humidity_at_2m:
        return file_relative_humidity_at_2m.variables["hurs"][:, :, :]

def UVtoSpeed(u, v):
    """Compute scalar wind speed (same units as input)."""
    return np.sqrt(u**2 + v**2)

def ProcessDailyWind(dir_u10, dir_v10, obs_hour=13):
    """
    Memory-safe conversion of hourly HCLIM3 uas/vas data into daily 13h wind speed & direction.
    Only keeps one hourly slice in memory per day.
    Returns:
        daily_speed [days, y, x],
    """
    with Dataset(dir_u10) as fu, Dataset(dir_v10) as fv:
        u_var = fu.variables["uas"]
        v_var = fv.variables["vas"]
        time_var = fu.variables["time"]

        calendar = getattr(time_var, "calendar", "standard")
        ts_cftime = num2date(time_var[:], units=time_var.units, calendar=calendar)
        timestamps = np.array(
            [
                datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                for t in ts_cftime
            ]
        )

        unique_days = np.unique([t.date() for t in timestamps])
        ny, nx = u_var.shape[1], u_var.shape[2]

        daily_speed = np.full((len(unique_days), ny, nx), np.nan, dtype=np.float32)
        daily_dir = np.full((len(unique_days), ny, nx), np.nan, dtype=np.float32)

        for i, day in enumerate(unique_days):
            idxs = np.where([t.date() == day for t in timestamps])[0]
            if idxs.size == 0:
                continue

            hours = np.array([timestamps[j].hour for j in idxs])
            idx_rel = np.argmin(np.abs(hours - obs_hour))
            idx_abs = idxs[idx_rel]

            u13 = u_var[idx_abs, :, :]
            v13 = v_var[idx_abs, :, :]

            daily_speed[i, :, :] = UVtoSpeed(u13, v13)
    return daily_speed

def TemperatureMax(dir_temperature_max):
    with Dataset(dir_temperature_max) as file_temperature_at_2m:
        temperature_max = file_temperature_at_2m.variables["tas"][:, :, :] - 273.15
        return temperature_max * (9 / 5) + 32 #Conversion Celsius to Fahrenheit

def TemperatureMin(dir_temperature_min, HCLIM3=False):
    with Dataset(dir_temperature_min) as file_temperature_at_2m:
        temperature_min = file_temperature_at_2m.variables["tas"][:, :, :] - 273.15
        return temperature_min * (9 / 5) + 32 #Conversion Celsius to Fahrenheit

def HumidityMax(dir_humidity_max, HCLIM3=False):
    with Dataset(dir_humidity_max) as file_relative_humidity_at_2m:
        return file_relative_humidity_at_2m.variables["hurs"][:, :, :]

def HumidityMin(dir_humidity_min, HCLIM3=False):
    with Dataset(dir_humidity_min) as file_relative_humidity_at_2m:
        return file_relative_humidity_at_2m.variables["hurs"][:, :, :]

def PrecipitationHours(dir_precipitation_annual, HCLIM3=False):
    with Dataset(dir_precipitation_annual) as file_precipitation_duration:
        return file_precipitation_duration.variables["count"][:, :, :]

def PrecipitationAnnual(dir_precipitation_annual, HCLIM3=False):
    with Dataset(dir_precipitation_annual) as file_precipitation_annual:
        return file_precipitation_annual.variables["pr"][0, :, :]

def DurationDaylight(latitude, day):
    phi = math.radians(latitude) #*0.01745 is degree to radian, so using builtin function instead
    solar_declination = 0.41008 * math.sin(math.radians((day + 1 - 82)))
    duration_daylight = 0
    if (math.tan(phi) * math.tan(solar_declination)) < -1:
        duration_daylight = 0
    elif (math.tan(phi) * math.tan(solar_declination)) > 1:
        duration_daylight = 24
    else:
        duration_daylight = 24 * (
            1 - (math.acos(math.tan(phi) * math.tan(solar_declination)) / 3.1416)
        )
    return duration_daylight



