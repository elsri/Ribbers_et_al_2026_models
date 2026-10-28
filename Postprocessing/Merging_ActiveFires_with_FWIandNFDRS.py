#!/usr/bin/env python

import xarray as xr
import pandas as pd
import numpy as np
import logging
import geopandas as gpd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(message)s")
logger = logging.getLogger(__name__)

def process_model_data(fire_shp_file, fwi_file, nfdr_file):
    logger.info("Opening datasets...")
    # Load shapefile instead of NetCDF
    gdf_fire = gpd.read_file(fire_shp_file)
    ds_fwi = xr.open_dataset(fwi_file)
    ds_nfdr = xr.open_dataset(nfdr_file)

    # 1. Filtering and Cleaning Shapefile Data
    logger.info("Filtering shapefile by confidence > 40...")
    # Standardize column names to lowercase for easier access if needed
    gdf_fire.columns = [c.lower() for c in gdf_fire.columns]
    
    # Apply filters
    df_fire = gdf_fire[gdf_fire['confidence'] > 40].copy()
    df_fire = df_fire[df_fire['frp'] > 0].copy()
    df_fire = df_fire.dropna(subset=['frp', 'latitude', 'longitude'])
    
    # Convert ACQ_DATE to datetime and create floor date for matching
    df_fire['time'] = pd.to_datetime(df_fire['acq_date'])
    df_fire['date'] = df_fire['time'].dt.floor('D')

    # 2. Time Range Alignment
    model_start = pd.to_datetime(ds_fwi.time.min().values)
    model_end = pd.to_datetime(ds_fwi.time.max().values)
    
    # Buffer the end by 13h to catch fire events on the last day 
    # (matching your original logic for the 11:30 vs 00:00 offset)
    df_fire = df_fire[
        (df_fire['time'] >= model_start) & 
        (df_fire['time'] <= model_end + pd.Timedelta('13h'))
    ]

    if df_fire.empty: 
        logger.warning("No fire events found within the weather model time range.")
        return None
    
    # Aggregate duplicate points on the same day/location
    df_fire = df_fire.groupby(['date', 'latitude', 'longitude'], as_index=False).agg({'frp': 'max'})

    # 3. Identify ONLY Land Points in the model
    logger.info("Identifying land points from model...")
    sample_slice = ds_fwi.fire_weather_index.isel(time=0).values
    land_mask = ~np.isnan(sample_slice)
    
    land_lats = ds_fwi.lat.values[land_mask]
    land_lons = ds_fwi.lon.values[land_mask]
    y_coords, x_coords = np.where(land_mask)

    # 4. Build Tree and Query
    logger.info("Building KDTree and querying nearest land pixels...")
    tree = cKDTree(np.c_[land_lats, land_lons])
    dist, indices = tree.query(np.c_[df_fire['latitude'], df_fire['longitude']])
    
    matched_y = y_coords[indices]
    matched_x = x_coords[indices]
    
    # Vectorized time matching
    time_values = ds_fwi.time.values
    time_indices = np.abs(
        time_values[:, None] - df_fire['date'].values[None, :]
    ).argmin(axis=0)
    
    # 5. Point-wise Sampling
    idx_t = xr.DataArray(time_indices, dims="event")
    idx_y = xr.DataArray(matched_y, dims="event")
    idx_x = xr.DataArray(matched_x, dims="event")

    logger.info("Sampling weather variables...")
    # Extract values directly from numpy arrays for speed
    df_fire['fire_weather_index'] = ds_fwi.fire_weather_index.values[idx_t, idx_y, idx_x]
    df_fire['initial_spread_index'] = ds_fwi.initial_spread_index.values[idx_t, idx_y, idx_x]
    df_fire['buildup_index'] = ds_fwi.buildup_index.values[idx_t, idx_y, idx_x]
    
    df_fire['burning_index'] = ds_nfdr.burning_index.values[idx_t, idx_y, idx_x]
    df_fire['energy_release_component'] = ds_nfdr.energy_release_component.values[idx_t, idx_y, idx_x]
    df_fire['spread_component'] = ds_nfdr.spread_component.values[idx_t, idx_y, idx_x]
    
    df_fire['dist_to_land_deg'] = dist

    ds_fwi.close(); ds_nfdr.close()
    return df_fire

# Paths
fire_radiative_power_file = "/PATH/TO/Fennoscandia_Active_Fires.shp"
fire_weather_index_file = "/PATH/TO/FWI_HCLIM3_Fennoscandia_2001_2018.nc"
burning_index_file = "/PATH/TO/NFDR_HCLIM3_Fennoscandia_2001_2018.nc"
output_csv = "/PATH/TO/active_fire_events_scandinavia.csv"

fire_pts = process_model_data(fire_radiative_power_file, fire_weather_index_file, burning_index_file)

if fire_pts is not None:
    logger.info("Saving to csv...")
    fire_pts.to_csv(output_csv, index=False)
    logger.info(f"Sample output saved. Total events: {len(fire_pts)}")
    print(fire_pts[['date', 'fire_weather_index', 'dist_to_land_deg']].head())
logger.info("Done.")
