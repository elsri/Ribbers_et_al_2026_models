#!/usr/bin/env python

import xarray as xr
import pandas as pd
import numpy as np
import logging
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(message)s")
logger = logging.getLogger(__name__)

def gridcell_area(lat):
    earth_radius = 6371000.0  # in meters
    degrees_lat = np.deg2rad(0.1)
    degrees_lon = np.deg2rad(0.1)
    lat_radius = np.deg2rad(lat)
    return ((earth_radius**2) * degrees_lat * degrees_lon * np.cos(lat_radius)) / 1000000

def process_model_data(frp_file, ba_file, fwi_file, nfdr_file):
    # ---------------------------------------------------------
    # 1. OPEN DATASETS
    # ---------------------------------------------------------

    logger.info("Opening datasets...")
    ds_ba = xr.open_dataset(ba_file, decode_times=True)
    ds_frp = xr.open_dataset(frp_file)
    ds_fwi = xr.open_dataset(fwi_file)
    ds_nfdr = xr.open_dataset(nfdr_file)

    #logger.info(f"GFED time dtype: {ds_ba.time.dtype}")
    #logger.info(f"GFED first times: {ds_ba.time.values[:5]}")
    #logger.info(f"GFED last times: {ds_ba.time.values[-5:]}")
    #logger.info(f"GFED time attrs: {ds_ba.time.attrs}")
    #logger.info(f"GFED time encoding: {ds_ba.time.encoding}")

    #logger.info("GFAS coordinates:")
    #logger.info(f"lat: {ds_frp.lat.values[:10]}")
    #logger.info(f"lon: {ds_frp.lon.values[:10]}")

    #logger.info("GFED coordinates:")
    #logger.info(f"lat: {ds_ba.lat.values[:10]}")
    #logger.info(f"lon: {ds_ba.lon.values[:10]}")

    #logger.info(f"GFAS shape: {ds_frp.sizes}")
    #logger.info(f"GFED shape: {ds_ba.sizes}")

    # ---------------------------------------------------------
    # 2. PREPARE GFAS FRP
    # ---------------------------------------------------------

    logger.info("Creating GFAS dataframe...")

    if ds_frp.lon.max() > 180:
        ds_frp = ds_frp.assign_coords(lon=(((ds_frp.lon + 180) % 360) - 180))
        ds_frp = ds_frp.sortby('lon')

    logger.info("Creating FRP dataframe with filtering above 0...")

    df_fire = (
        ds_frp
        .to_dataframe()
        .reset_index()
    )

    df_fire = df_fire.dropna(subset=["frpfire"])
    df_fire = df_fire[df_fire["frpfire"] > 0]

    # Preserve the original GFAS timestamp
    df_fire["time"] = pd.to_datetime(df_fire["time"])

    # Create monthly key for matching to GFED
    df_fire["month"] = df_fire["time"].dt.to_period("M")

    logger.info(
        f"Positive GFAS observations: {len(df_fire)}"
    )

    # ---------------------------------------------------------
    # 3. SELECTING ONE FRP FIRE PER GRID CELL PER MONTH
    # ---------------------------------------------------------

    logger.info(
        "Selecting maximum FRP per grid cell and month..."
    )

    # Find the index of the maximum FRP in each
    # lat/lon/month group
    max_idx = (
        df_fire
        .groupby(["lat", "lon", "month"])["frpfire"]
        .idxmax()
    )

    # Keep the complete original GFAS record corresponding
    # to the maximum FRP, including its timestamp
    df_fire = (
        df_fire
        .loc[max_idx]
        .reset_index(drop=True)
    )

    # Keep original GFAS timestamp
    df_fire["time"] = pd.to_datetime(df_fire["time"])

    # Daily date used only for matching FWI/NFDRS
    df_fire["date"] = df_fire["time"].dt.floor("D")

    # Monthly key used only for matching GFED
    df_fire["month"] = df_fire["time"].dt.to_period("M")

    df_fire["gridcell_area_m2"] = gridcell_area(df_fire["lat"])
    df_fire["frp_megawatts"] = df_fire["frpfire"] * df_fire["gridcell_area_m2"]

    logger.info(
        f"Unique fire gridcell-month events: {len(df_fire)}"
    )

    # --------------------------------------------------
    # 4. MATCH GFAS EVENTS TO GFED5 BURNED AREA
    # --------------------------------------------------

    df_burned = (
        ds_ba
        .to_dataframe()
        .reset_index()
    )

    df_burned["time"] = pd.to_datetime(df_burned["time"])
    df_burned["month"] = df_burned["time"].dt.to_period("M")

    # Only positive burned area is relevant
    df_burned = df_burned.dropna(subset=["burned_area"])
    df_burned = df_burned[df_burned["burned_area"] > 0]

    logger.info(f"GFED rows with BA > 0: {len(df_burned)}")

    # --------------------------------------------------
    # Find nearest GFED grid cell
    # --------------------------------------------------

    gfed_grid = (
        df_burned[["lat", "lon"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    gfed_tree = cKDTree(
        gfed_grid[["lat", "lon"]].values
    )

    dist_gfed, idx_gfed = gfed_tree.query(
        df_fire[["lat", "lon"]].values
    )

    df_fire["gfed_lat"] = gfed_grid.loc[idx_gfed, "lat"].values
    df_fire["gfed_lon"] = gfed_grid.loc[idx_gfed, "lon"].values
    df_fire["dist_to_gfed_deg"] = dist_gfed

    # --------------------------------------------------
    # Match GFED month, allowing for a month detection delay
    #
    # Priority:
    #   1. Same month
    #   2. One month earlier
    # --------------------------------------------------

    logger.info("Matching GFAS events to GFED burned area...")

    # Give each GFAS event a unique ID
    df_fire = df_fire.reset_index(drop=True)
    df_fire["event_id"] = np.arange(len(df_fire))

    # Create candidate GFED months
    df_fire["gfed_month_same"] = df_fire["month"]
    df_fire["gfed_month_previous"] = (
        df_fire["month"] - 1
    )

    # GFED lookup table
    gfed_lookup = df_burned[
        ["lat", "lon", "month", "burned_area"]
    ].rename(
        columns={
            "lat": "gfed_lat",
            "lon": "gfed_lon"
        }
    )

    # --------------------------------------------------
    # 1. Try SAME month
    # --------------------------------------------------

    same_month = df_fire[
        ["event_id", "gfed_lat", "gfed_lon", "gfed_month_same"]
    ].rename(
        columns={
            "gfed_month_same": "month"
        }
    )

    same_month = same_month.merge(
        gfed_lookup,
        on=["gfed_lat", "gfed_lon", "month"],
        how="inner"
    )

    same_month["gfed_match_type"] = "same_month"
    same_month["gfed_month"] = same_month["month"]

    logger.info(
        f"Events matched in same month: {len(same_month)}"
    )

    # --------------------------------------------------
    # 2. Identify events NOT matched in same month
    # --------------------------------------------------

    matched_same_ids = set(same_month["event_id"])

    previous_candidates = df_fire[
        ~df_fire["event_id"].isin(matched_same_ids)
    ][
        ["event_id", "gfed_lat", "gfed_lon", "gfed_month_previous"]
    ].rename(
        columns={
            "gfed_month_previous": "month"
        }
    )

    # --------------------------------------------------
    # 3. Try ONE MONTH EARLIER
    # --------------------------------------------------

    previous_month = previous_candidates.merge(
        gfed_lookup,
        on=["gfed_lat", "gfed_lon", "month"],
        how="inner"
    )

    previous_month["gfed_match_type"] = "one_month_earlier"
    previous_month["gfed_month"] = previous_month["month"]

    logger.info(
        f"Events matched one month earlier: "
        f"{len(previous_month)}"
    )

    # --------------------------------------------------
    # 4. Combine matches
    # --------------------------------------------------

    matched = pd.concat(
        [same_month, previous_month],
        ignore_index=True
    )

    logger.info(
        f"Total GFAS events matched to GFED BA: "
        f"{len(matched)}"
    )

    logger.info(
        f"GFAS events with no GFED match: "
        f"{len(df_fire) - len(matched)}"
    )

    # --------------------------------------------------
    # 5. Add GFED result back to original GFAS events
    # --------------------------------------------------

    df_fire = df_fire.merge(
        matched[
            [
                "event_id",
                "burned_area",
                "gfed_month",
                "gfed_match_type"
            ]
        ],
        on="event_id",
        how="inner"
    )

    # --------------------------------------------------
    # 6. Calculate FRP / burned area
    # --------------------------------------------------

    df_fire["frp_per_burned_area"] = (
        df_fire["frp_megawatts"] /
        df_fire["burned_area"]
    )
    df_fire["combined_fire_danger_frp_times_ba"] = (
        df_fire["frp_megawatts"] *
        df_fire["burned_area"]        
    )

    logger.info(
        f"Final GFAS-GFED events: {len(df_fire)}"
    )

    logger.info(
        "GFED match type counts:\n"
        + str(df_fire["gfed_match_type"].value_counts())
    )

    df_fire["date"] = df_fire["time"].dt.floor("D")
    logger.info(
        f"GFAS-GFED matching distance: "
        f"min={df_fire['dist_to_gfed_deg'].min():.4f}, "
        f"max={df_fire['dist_to_gfed_deg'].max():.4f}, "
        f"mean={df_fire['dist_to_gfed_deg'].mean():.4f} degrees"
    )

    # --------------------------------------------------
    # 5. MATCHING WITH FIRE DANGER MODELS
    # --------------------------------------------------

    model_start = max(ds_fwi.time.min().values, ds_nfdr.time.min().values)
    model_end = min(ds_fwi.time.max().values, ds_nfdr.time.max().values)
    
    # Buffer the end by 1 day because fire 'date' is floored to 00:00:00
    # and the model is at ~11:00:00. 
    df_fire = df_fire[
        (df_fire['time'] >= model_start) & 
        (df_fire['time'] <= pd.to_datetime(model_end) + pd.Timedelta('13h'))
    ]

    if df_fire.empty: 
        logger.warning("No fire events found within the weather model time range.")
        return None
    
    # 2. Identify ONLY Land Points in the model
    # We use the first time step of FWI to find where data actually exists
    logger.info("Indentifying land points...")
    sample_slice = ds_fwi.fire_weather_index.isel(time=0).values
    land_mask = ~np.isnan(sample_slice)
    
    # Get coordinates of land-only points
    land_lats = ds_fwi.lat.values[land_mask]
    land_lons = ds_fwi.lon.values[land_mask]
    # Keep track of their original 2D indices
    y_coords, x_coords = np.where(land_mask)

    # 3. Build Tree using ONLY Land Points
    logger.info("Building KDTree from land-only pixels...")
    tree = cKDTree(np.c_[land_lats, land_lons])

    # Query: This will now find the nearest pixel that DEFINITELY has data
    dist, indices = tree.query(np.c_[df_fire['lat'], df_fire['lon']])
    
    # Map back to the original Y, X of the full model grid
    matched_y = y_coords[indices]
    matched_x = x_coords[indices]
    time_indices = np.abs(
        ds_fwi.time.values[:, None] - df_fire['date'].values[None, :]
    ).argmin(axis=0)
    
    # 4. Point-wise Selection
    idx_t = xr.DataArray(time_indices, dims="event")
    idx_y = xr.DataArray(matched_y, dims="event")
    idx_x = xr.DataArray(matched_x, dims="event")

    logger.info("Sampling data from nearest land pixels...")

    fwi_raw = ds_fwi.fire_weather_index.values[idx_t, idx_y, idx_x]
    isi_raw = ds_fwi.initial_spread_index.values[idx_t, idx_y, idx_x]
    bui_raw = ds_fwi.buildup_index.values[idx_t, idx_y, idx_x]
    bi_raw = ds_nfdr.burning_index.values[idx_t, idx_y, idx_x]
    erc_raw = ds_nfdr.energy_release_component.values[idx_t, idx_y, idx_x]
    sc_raw = ds_nfdr.spread_component.values[idx_t, idx_y, idx_x]

    # 5. Assignment
    df_fire['fire_weather_index'] = fwi_raw
    df_fire['initial_spread_index'] = isi_raw
    df_fire['buildup_index'] = bui_raw
    df_fire['burning_index'] = bi_raw
    df_fire['energy_release_component'] = erc_raw
    df_fire['spread_component'] = sc_raw
    df_fire['dist_to_land_deg'] = dist

    ds_frp.close(); ds_fwi.close(); ds_nfdr.close()
    return df_fire

# Paths (Keep your existing paths)
fire_radiative_power_file = "/PATH/TO/gfas_Fennoscandia_2003_2024.nc"
burned_area_file = "/PATH/TO/gfed5_Fennoscandia_2003_2018_fixed.nc"
fire_weather_index_file = "/PATH/TO/FWI_HCLIM3_Fennoscandia_2001_2018.nc"
burning_index_file = "/PATH/TO/NFDR_HCLIM3_Fennoscandia_2001_2018.nc"
output_csv = "/PATH/TO/gfas_gfed_events_scandinavia.csv"

fire_pts = process_model_data(frp_file=fire_radiative_power_file, ba_file=burned_area_file, fwi_file=fire_weather_index_file, nfdr_file=burning_index_file)
logger.info("Saving to csv...")
if fire_pts is not None:
    fire_pts.to_csv(output_csv, index=False)
    logger.info("Sample output:")
    logger.info(fire_pts[['date', 'fire_weather_index', 'dist_to_land_deg']].head())
logger.info("Done.")
