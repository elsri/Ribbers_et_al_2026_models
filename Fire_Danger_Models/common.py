import numpy as np
from netCDF4 import Dataset, num2date
from numpy.ma.core import MaskedArray
import math
import pandas as pd
from datetime import datetime


def Precipitation(dir_precipitation, HCLIM3=False, CARRA2=False):
    with Dataset(dir_precipitation) as file_precipitation:
        if HCLIM3:
            return file_precipitation.variables["pr"][:, :, :]
        elif CARRA2:
            return file_precipitation.variables["tp"][:, :, :]
        return file_precipitation.variables["RAINNC"][:, :, :]


def Windspeeds(dir_windspeeds, HCLIM3=False, CARRA2=False):
    with Dataset(dir_windspeeds) as file_windspeeds:
        if CARRA2:
            return file_windspeeds.variables["si10"][:, :, :]


def Temperature(dir_temperature, HCLIM3=False, CARRA2=False):
    with Dataset(dir_temperature) as file_temperature_at_2m:
        if HCLIM3:
            return file_temperature_at_2m.variables["tas"][:, :, :] - 273.15
        elif CARRA2:
            return file_temperature_at_2m.variables["t2m"][:, :, :] - 273.15
        return file_temperature_at_2m.variables["T2MEAN"][:, :, :] - 273.15


def TemperatureMax(dir_temperature, HCLIM3=False, CARRA2=False):
    with Dataset(dir_temperature) as file_temperature_at_2m:
        if HCLIM3:
            return file_temperature_at_2m.variables["tas"][:, :, :] - 273.15
        elif CARRA2:
            return file_temperature_at_2m.variables["t2m"][:, :, :] - 273.15
        return file_temperature_at_2m.variables["T2MEAN"][:, :, :] - 273.15


def TemperatureMin(dir_temperature, HCLIM3=False, CARRA2=False):
    with Dataset(dir_temperature) as file_temperature_at_2m:
        if HCLIM3:
            return file_temperature_at_2m.variables["tas"][:, :, :] - 273.15
        elif CARRA2:
            return file_temperature_at_2m.variables["t2m"][:, :, :] - 273.15
        return file_temperature_at_2m.variables["T2MEAN"][:, :, :] - 273.15


def HumidityMean(dir_humidity, HCLIM3=False, CARRA2=False):
    with Dataset(dir_humidity) as file_relative_humidity_at_2m:
        if HCLIM3:
            return file_relative_humidity_at_2m.variables["hurs"][:, :, :]
        elif CARRA2:
            return file_relative_humidity_at_2m.variables["r2"][:, :, :]
        return file_relative_humidity_at_2m.variables["HURS"][:, :, :] * 100


def HumidityMax(dir_humidity, HCLIM3=False, CARRA2=False):
    with Dataset(dir_humidity) as file_relative_humidity_at_2m:
        if HCLIM3:
            return file_relative_humidity_at_2m.variables["hurs"][:, :, :]
        elif CARRA2:
            return file_relative_humidity_at_2m.variables["r2"][:, :, :]
        return file_relative_humidity_at_2m.variables["HURS"][:, :, :] * 100


def HumidityMin(dir_humidity, HCLIM3=False, CARRA2=False):
    with Dataset(dir_humidity) as file_relative_humidity_at_2m:
        if HCLIM3:
            return file_relative_humidity_at_2m.variables["hurs"][:, :, :]
        elif CARRA2:
            return file_relative_humidity_at_2m.variables["r2"][:, :, :]
        return file_relative_humidity_at_2m.variables["HURS"][:, :, :] * 100


def Snow(dir_snow, HCLIM3=False, CARRA2=False):
    with Dataset(dir_snow) as file_snow_ratio:
        if HCLIM3:
            return file_snow_ratio.variables["snc"][:, :, :]
        elif CARRA2:
            return file_snow_ratio.variables["r2"][:, :, :]
        return file_snow_ratio.variables["snow"][:, :, :]


def SolarRadiation(dir_solar, HCLIM3=False, CARRA2=False):
    with Dataset(dir_solar) as file_solar_radiation:
        if HCLIM3:
            return file_solar_radiation.variables["rsds"][:, :, :]
        elif CARRA2:
            return file_solar_radiation.variables["r2"][:, :, :]
        return file_solar_radiation.variables["rsds"][:, :, :]


def Time(dir_time, HCLIM3=False, CARRA2=False):
    with Dataset(dir_time) as file_time:
        time_var = file_time.variables["time"]
        if HCLIM3:
            return time_var[:], time_var.units

def uv_to_speed(u, v):
    """Compute scalar wind speed (same units as input)."""
    return np.sqrt(u**2 + v**2)


def uv_to_dir_deg(u, v):
    """
    Convert U, V to meteorological wind direction (0–360°, from which wind blows).
    0° = from North, 90° = from East, etc.
    """
    return (180 + (180 / math.pi) * np.arctan2(v, u)) % 360


def uv_to_daily13h(u10, v10, timestamps, obs_hour=13):
    """Return daily wind speed and direction at ~13:00 local."""
    # ts = pd.DatetimeIndex(timestamps)
    ts = pd.to_datetime(timestamps)
    # print([type(t) for t in ts[:5]])
    print("len(ts):", len(ts))
    print("u10.shape:", u10.shape)

    valid = ~pd.isna(ts)
    ts = ts[valid]
    u10 = u10[valid]
    v10 = v10[valid]

    ts_values = ts.to_numpy()

    # df = pd.DataFrame({'ts': ts})
    # df['day'] = df['ts'].dt.floor('D')

    # unique_days = np.unique(ts.date)
    unique_days = np.unique(ts_values.astype("datetime64[D]"))
    n_days = len(unique_days)
    ny, nx = u10.shape[1], u10.shape[2]

    daily_speed = np.full((n_days, ny, nx), np.nan, dtype=np.float32)
    daily_dir = np.full((n_days, ny, nx), np.nan, dtype=np.float32)

    print("First timestamp:", ts[0])
    print("Last timestamp:", ts[-1])
    print("Unique days:", len(np.unique(ts.date)))
    print("Any NaT?", pd.isna(ts).any())

    for i, day in enumerate(unique_days):
        # mask = (ts.date == day)
        # mask = (df['day'] == day).values
        mask = ts_values.astype("datetime64[D]") == day
        if not np.any(mask):
            continue

        ts_day = ts_values[mask]
        if len(ts_day) == 0:
            continue

        # hours = np.array([t.hour for t in ts_day])
        # hours = np.array([t.hour + t.minute/60 for t in ts_day])
        hours = np.array([t.astype("datetime64[h]").astype(int) % 24 for t in ts_day])
        idx = np.argmin(np.abs(hours - obs_hour))  # closest to 13h local

        time_index = np.nonzero(mask)[0][idx]
        u13 = u10[mask][time_index, :, :]
        v13 = v10[mask][time_index, :, :]

        daily_speed[i, :, :] = uv_to_speed(u13, v13)
        daily_dir[i, :, :] = uv_to_dir_deg(u13, v13)

    return daily_speed, daily_dir, unique_days


def uv_to_daily_gust(ugsm, vgsm, timestamps):
    """Compute daily maximum gust magnitude and direction."""
    ts = pd.DatetimeIndex(timestamps)
    print([type(t) for t in ts[:5]])
    print("len(ts):", len(ts))
    print("u10.shape:", ugsm.shape)

    unique_days = np.unique(ts.date)
    n_days = len(unique_days)
    ny, nx = ugsm.shape[1], ugsm.shape[2]

    daily_gust_speed = np.full((n_days, ny, nx), np.nan)
    daily_gust_dir = np.full((n_days, ny, nx), np.nan)

    for i, day in enumerate(unique_days):
        mask = ts.date == day
        if not np.any(mask):
            continue

        u_day = ugsm[mask, :, :]
        v_day = vgsm[mask, :, :]
        gust_speed = uv_to_speed(u_day, v_day)
        idx = np.nanargmax(gust_speed, axis=0)
        gust_speed_max = np.nanmax(gust_speed, axis=0)

        # direction at the same index as the gust maximum
        gust_dir = np.empty_like(gust_speed_max)
        for iy in range(ny):
            for ix in range(nx):
                gust_dir[iy, ix] = uv_to_dir_deg(
                    u_day[idx[iy, ix], iy, ix],
                    v_day[idx[iy, ix], iy, ix],
                )

        daily_gust_speed[i, :, :] = gust_speed_max
        daily_gust_dir[i, :, :] = gust_dir

    return daily_gust_speed, daily_gust_dir, unique_days

def process_daily_wind_HCLIM3(dir_u10, dir_v10, obs_hour=13):
    """
    Memory-safe conversion of hourly HCLIM3 uas/vas data into daily 13h wind speed & direction.
    Only keeps one hourly slice in memory per day.
    Returns:
        daily_speed [days, y, x],
        daily_dir [days, y, x],
        wind_time [array of datetime64 days]
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

        # print(f"[Wind] Processing {len(unique_days)} days × {ny} × {nx} from {dir_u10}")

        for i, day in enumerate(unique_days):
            idxs = np.where([t.date() == day for t in timestamps])[0]
            if idxs.size == 0:
                continue

            hours = np.array([timestamps[j].hour for j in idxs])
            idx_rel = np.argmin(np.abs(hours - obs_hour))
            idx_abs = idxs[idx_rel]

            u13 = u_var[idx_abs, :, :]
            v13 = v_var[idx_abs, :, :]

            daily_speed[i, :, :] = uv_to_speed(u13, v13)
            daily_dir[i, :, :] = uv_to_dir_deg(u13, v13)

        wind_time = np.array([np.datetime64(pd.Timestamp(d)) for d in unique_days])

    return daily_speed, daily_dir, wind_time


def process_daily_gust_HCLIM3(dir_ugsm, dir_vgsm, obs_hour=13):
    """
    Memory-safe conversion of hourly HCLIM3 gust components into daily 13h gust speed & direction.
    Returns:
        daily_gust_speed [days, y, x],
        daily_gust_dir [days, y, x],
        gust_time [array of datetime64 days]
    """
    with Dataset(dir_ugsm) as fu, Dataset(dir_vgsm) as fv:
        u_var = fu.variables["ugsm"]
        v_var = fv.variables["vgsm"]
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

        daily_gust_speed = np.full((len(unique_days), ny, nx), np.nan, dtype=np.float32)
        daily_gust_dir = np.full((len(unique_days), ny, nx), np.nan, dtype=np.float32)

        print(
            f"[Gust] Processing {len(unique_days)} days × {ny} × {nx} from {dir_ugsm}"
        )

        for i, day in enumerate(unique_days):
            idxs = np.where([t.date() == day for t in timestamps])[0]
            if idxs.size == 0:
                continue

            hours = np.array([timestamps[j].hour for j in idxs])
            idx_rel = np.argmin(np.abs(hours - obs_hour))
            idx_abs = idxs[idx_rel]

            u13 = u_var[idx_abs, :, :]
            v13 = v_var[idx_abs, :, :]

            daily_gust_speed[i, :, :] = uv_to_speed(u13, v13)
            daily_gust_dir[i, :, :] = uv_to_dir_deg(u13, v13)

        gust_time = np.array([np.datetime64(pd.Timestamp(d)) for d in unique_days])

    return daily_gust_speed, daily_gust_dir, gust_time


def Slope(
    index_y,
    index_x,
    number_of_latitudinal_degrees,
    number_of_longitudinal_degrees,
    elevation,
):
    if (
        index_y > 0
        and index_x > 0
        and index_y < number_of_latitudinal_degrees - 1
        and index_x < number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
            abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
        ]
    elif (
        index_y == 0
        and index_x > 0
        and index_y < number_of_latitudinal_degrees - 1
        and index_x < number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
            abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
        ]
    elif (
        index_y > 0
        and index_x == 0
        and index_y < number_of_latitudinal_degrees - 1
        and index_x < number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
        ]
    elif (
        index_y > 0
        and index_x > 0
        and index_y == number_of_latitudinal_degrees - 1
        and index_x < number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
            abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
        ]
    elif (
        index_y > 0
        and index_x > 0
        and index_y < number_of_latitudinal_degrees - 1
        and index_x == number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
            abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
        ]
    elif (
        index_y == 0
        and index_x == 0
        and index_y < number_of_latitudinal_degrees - 1
        and index_x < number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
        ]
    elif (
        index_y > 0
        and index_x > 0
        and index_y == number_of_latitudinal_degrees - 1
        and index_x == number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
            abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
        ]
    elif (
        index_y == 0
        and index_x > 0
        and index_y < number_of_latitudinal_degrees - 1
        and index_x == number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
            abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
        ]
    elif (
        index_y > 0
        and index_x == 0
        and index_y == number_of_latitudinal_degrees - 1
        and index_x < number_of_longitudinal_degrees - 1
    ):
        elevations = [
            abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
            abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
        ]

    slope = math.atan((max(elevations)) / 3000)
    return slope
