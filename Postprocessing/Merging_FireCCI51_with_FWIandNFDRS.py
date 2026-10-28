#!/usr/bin/env python

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
import logging
import gc
from shapely.geometry import box
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. Functions definition
# ---------------------------------------------------------------------
def split_by_time_gap(group, max_gap_days=4):
    """
    Splits a spatial patch into multiple events if there is a 
    time gap of more than max_gap_days between detections.
    """
    group = group.sort_values("ACQ_DATE")
    # Calculate days between consecutive detections in this spatial patch
    group['date_diff'] = group['ACQ_DATE'].diff().dt.days.fillna(0)
    # Create a sub-id that increments whenever the gap is exceeded
    group['sub_event_id'] = (group['date_diff'] > max_gap_days).cumsum()
    return group

def ensure_lat_ascending(ds):
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])
    return ds

def process_model_data(model_path, var_name, event_df, prefix):
    logger.info(f"Processing {prefix} ({var_name}) with vectorized extraction...")
    ds = xr.open_dataset(model_path, chunks={'time': -1})
    
    # --- 1. SPATIAL PREP ---
    sample_slice = ds[var_name].isel(time=0).load()
    valid_mask = np.isfinite(sample_slice.values)
    lats, lons = ds.lat.values, ds.lon.values
    tree_coords = np.stack([lats[valid_mask], lons[valid_mask]], axis=1)
    y_indices, x_indices = np.where(valid_mask)
    tree = cKDTree(tree_coords)
    
    # Collect every single lat/lon point from all patches to find required cells
    all_patch_lats = [lat for sublist in event_df['lat'] for lat in sublist]
    all_patch_lons = [lon for sublist in event_df['lon'] for lon in sublist]
    global_coords = np.column_stack([all_patch_lats, all_patch_lons])
    
    _, idxs = tree.query(global_coords, k=1, distance_upper_bound=0.2)
    valid_mask_global = idxs < len(tree_coords)
    
    # Identify unique weather grid cells (Y, X) needed for the whole run
    unique_yx = pd.DataFrame({
        'y': y_indices[idxs[valid_mask_global]],
        'x': x_indices[idxs[valid_mask_global]]
    }).drop_duplicates()
    
    # --- 2. THE LOOKUP & MEGA-LOAD ---
    # We define 'lookup' here so the loop knows which column belongs to which (y, x)
    lookup = {(r.y, r.x): i for i, r in enumerate(unique_yx.itertuples(index=False))}
    
    logger.info(f"Loading {len(unique_yx)} weather cells for all events into RAM...")
    subset = ds[var_name].isel(
        y=xr.DataArray(unique_yx.y.values, dims="pts"),
        x=xr.DataArray(unique_yx.x.values, dims="pts")
    ).load()
    
    subset_vals = subset.values 
    time_axis = pd.to_datetime(ds.time.values)
    
    # --- 3. COMPUTE METRICS ---
    # Pre-convert dates to numpy to avoid Pandas Series indexing issues
    starts = pd.to_datetime(event_df['Start_date'].values).values
    ends = pd.to_datetime(event_df['End_date'].values).values
    
    means, maxes = [], []
    for i in range(len(event_df)):
        patch_coords = np.column_stack([event_df['lat'].iloc[i], event_df['lon'].iloc[i]])
        _, p_idxs = tree.query(patch_coords, k=1, distance_upper_bound=0.2)
        valid_p = p_idxs[p_idxs < len(tree_coords)]
        
        # Now 'lookup' is defined and we can get the column pointers
        ptrs = list({lookup.get((y_indices[v], x_indices[v])) for v in valid_p 
                     if lookup.get((y_indices[v], x_indices[v])) is not None})
        
        if ptrs:
            t_start = int(np.searchsorted(time_axis, starts[i]))
            t_end = int(np.searchsorted(time_axis, ends[i]))
            
            # Slice time window, then take specific patch columns
            window = np.take(subset_vals[t_start : t_end + 1, :], ptrs, axis=1)
            
            if window.size > 0 and np.any(np.isfinite(window)):
                means.append(np.nanmean(window))
                maxes.append(np.nanmax(window))
            else:
                means.append(np.nan); maxes.append(np.nan)
        else:
            means.append(np.nan); maxes.append(np.nan)
            
    event_df[f"{prefix}_mean"] = means
    event_df[f"{prefix}_max"] = maxes
    
    del ds, subset, subset_vals
    gc.collect()
    return event_df

def make_box(row):
    return box(row.lon - res_lon/2, row.lat - res_lat/2, 
               row.lon + res_lon/2, row.lat + res_lat/2)
    
def cluster_burned_pixels(group, dist_threshold=0.35):
    """
    Groups the MODIS grid cells themselves. 
    A threshold of 0.35 degrees is slightly larger than one 0.25 deg pixel, 
    ensuring diagonal or adjacent pixels are linked.
    """
    if len(group) < 2:
        group['Patch_ID'] = 0
        return group
    
    # We only care about unique BA grid cells for the clustering math
    unique_pixels = group.drop_duplicates(subset=['ba_lat', 'ba_lon']).copy()
    coords = unique_pixels[['ba_lat', 'ba_lon']].values
    
    dist_matrix = squareform(pdist(coords))
    adj_matrix = dist_matrix < dist_threshold
    n_components, labels = connected_components(adj_matrix, directed=False)
    
    unique_pixels['Patch_ID'] = labels
    
    # Map these Patch IDs back to every active fire detection in the main group
    pixel_to_patch = unique_pixels.set_index(['ba_lat', 'ba_lon'])['Patch_ID'].to_dict()
    group['Patch_ID'] = group.set_index(['ba_lat', 'ba_lon']).index.map(pixel_to_patch)
    
    return group

def aggregate_border_blind(group):
    fire_dates = pd.to_datetime(group['ACQ_DATE'])
    duration = (fire_dates.max() - fire_dates.min()).days
    
    # Identify unique burned area grid cells associated with this specific temporal pulse
    unique_ba_data = group.drop_duplicates(subset=['ba_lat', 'ba_lon'])
    
    return pd.Series({
        'Start_date': fire_dates.min(),
        'End_date': fire_dates.max(),
        'Duration': duration,
        'Confidence': group['CONFIDENCE'].max(),
        'Fire_radiative_power_max': group['FRP'].max(),
        'Fire_radiative_power_mean': group['FRP'].mean(),
        'lat': list(group['lat']),
        'lon': list(group['lon']),
        'Burned_area': (unique_ba_data['Burned_area'].sum()),
        'lat_final': unique_ba_data['ba_lat'].mean(),
        'lon_final': unique_ba_data['ba_lon'].mean(),
    })

# ---------------------------------------------------------------------
# 2. USER INPUTS
# ---------------------------------------------------------------------
active_fire_shp = "/PATH/TO/Fennoscandia_Active_Fires.shp"
burned_file = "/PATH/TO/MODIS_2001_2018_Fennoscandia.nc"
muni_shapefile = "/PATH/TO/ScandinaviaKommuner.shp"
fwi_file = "//PATH/TO/FWI_HCLIM3_Fennoscandia_2001_2018.nc"
bi_file = "/PATH/TO/NFDR_HCLIM3_Fennoscandia_2001_2018.nc"
output_csv = "/PATH/TO/fire_events_final_scandinavia.csv"

# ---------------------------------------------------------------------
# 3. PROCESSING BURNED AREA AND ACTIVE FIRES
# ---------------------------------------------------------------------
logger.info("Loading Burned Area data...")
burned = ensure_lat_ascending(xr.open_dataset(burned_file))
ba_masked = burned["burned_area"].where(burned["burned_area"] > 0, drop=True)
ba_df = ba_masked.to_dataframe().dropna(subset=["burned_area"]).reset_index()
ba_df = ba_df.rename(columns={"burned_area": "Burned_area"})

# Determine grid resolution for polygon creation (usually 0.25 for GFED/MODIS)
res_lat = np.abs(np.diff(burned.lat.values[:2])[0])
res_lon = np.abs(np.diff(burned.lon.values[:2])[0])

ba_gdf = gpd.GeoDataFrame(
    ba_df, 
    geometry=ba_df.apply(make_box, axis=1), 
    crs="EPSG:4326"
)
ba_gdf["month_year"] = pd.to_datetime(ba_gdf["time"]).dt.to_period("M")

ba_gdf["ba_lat"] = ba_gdf["lat"]
ba_gdf["ba_lon"] = ba_gdf["lon"]

logger.info("Loading Active Fire shapefile...")
fire_pts = gpd.read_file(active_fire_shp)
if fire_pts.crs != "EPSG:4326":
    fire_pts = fire_pts.to_crs("EPSG:4326")

fire_pts["month_year"] = pd.to_datetime(fire_pts["ACQ_DATE"]).dt.to_period("M")

# ---------------------------------------------------------------------
# 4. SPATIAL JOINS
# ---------------------------------------------------------------------
logger.info("Performing spatial join: Fires -> Burned Boxes...")
ba_gdf["prev_month"] = (ba_gdf["time"] - pd.DateOffset(months=1)).dt.to_period("M")

# Inner join
matched_fires = gpd.sjoin(fire_pts, ba_gdf, how="inner", predicate="within")
matched_fires = matched_fires.drop(columns=['index_right'], errors='ignore')

# Temporal filtering
date_mask = (matched_fires["month_year_left"] == matched_fires["month_year_right"]) | \
            (matched_fires["month_year_left"] == matched_fires["prev_month"])
matched_fires = matched_fires[date_mask].copy()

logger.info("Performing spatial join: Fires -> Municipalities...")
muni = gpd.read_file(muni_shapefile)

if muni.crs != matched_fires.crs:
    muni = muni.to_crs(matched_fires.crs)
matched_fires = gpd.sjoin(matched_fires, muni[["geometry", "Municipali"]], how="inner", predicate="within")

matched_fires = matched_fires.drop(columns=['index_right'], errors='ignore')

# -----------------------------------------------------------------
# 5. PATCH GROUPING
# -----------------------------------------------------------------
logger.info("Grouping pixels into patches and handling temporal outliers...")

matched_fires = matched_fires.groupby(['month_year_right']).apply(
    cluster_burned_pixels, include_groups=False
).reset_index()

matched_fires = matched_fires.groupby(['month_year_right', 'Patch_ID']).apply(
    lambda x: split_by_time_gap(x, max_gap_days=4), include_groups=False
).reset_index()

matched_fires['Global_Patch_ID'] = (
    matched_fires['month_year_right'].astype(str) + "_" + 
    matched_fires['Patch_ID'].astype(str) + "_" +
    matched_fires['sub_event_id'].astype(str)
)

events = matched_fires.groupby('Global_Patch_ID').apply(
    aggregate_border_blind, include_groups=False
).reset_index()
events = events.rename(columns={"Lat_final": "Lat", "Lon_final": "Lon"})

logger.info(f"Identified {len(events)} patches. Starting model extraction")

# ---------------------------------------------------------------------
# 6. TARGETED MODEL EXTRACTION
# ---------------------------------------------------------------------
events = events.rename(columns={"lat_final": "Latitude", "lon_final": "Longitude"})
events['Start_date'] = pd.to_datetime(events['Start_date'])
events['End_date'] = pd.to_datetime(events['End_date'])
events["Latitude"] = events["lat"].apply(np.mean)
events["Longitude"] = events["lon"].apply(np.mean)

events = process_model_data(fwi_file, "fire_weather_index", events, "Fire_weather_index")
events = process_model_data(fwi_file, "initial_spread_index", events, "Initial_spread_index")
events = process_model_data(fwi_file, "buildup_index", events, "Buildup_index")
events = process_model_data(fwi_file, "starting_fine_fuel_moisture_code", events, "Starting_fine_fuel_moisture_code")
events = process_model_data(fwi_file, "starting_duff_moisture_code", events, "Starting_duff_moisture_code")
events = process_model_data(fwi_file, "starting_drought_code", events, "Starting_drought_code")

events = process_model_data(bi_file, "burning_index", events, "Burning_index")
events = process_model_data(bi_file, "spread_component", events, "Spread_component")
events = process_model_data(bi_file, "energy_release_component", events, "Energy_release_component")
events = process_model_data(bi_file, "moisture_content_herbaceous", events, "Moisture_content_herbaceous")
events = process_model_data(bi_file, "moisture_content_woody", events, "Moisture_content_woody")
events = process_model_data(bi_file, "dead_one_hour_fuel_moisture", events, "Dead_one_hour_fuel_moisture")
events = process_model_data(bi_file, "dead_ten_hour_fuel_moisture", events, "Dead_ten_hour_fuel_moisture")
events = process_model_data(bi_file, "dead_hundred_hour_fuel_moisture", events, "Dead_hundred_hour_fuel_moisture")
events = process_model_data(bi_file, "dead_thousand_hour_fuel_moisture", events, "Dead_thousand_hour_fuel_moisture")

id_cols = ['Latitude', 'Longitude', 'Start_date', 'End_date']

aggregation_rules = {
    'Burned_area': 'max',
    'Fire_radiative_power_max': 'max',
    'Fire_radiative_power_mean': 'mean',
    'Confidence': 'max',
    'Duration': 'first',
    'Fire_weather_index_mean': 'mean',
    'Fire_weather_index_max': 'max',
    'Initial_spread_index_mean': 'mean',
    'Initial_spread_index_max': 'max',
    'Buildup_index_mean': 'mean',
    'Buildup_index_max': 'max',
    'Starting_fine_fuel_moisture_code_mean': 'mean',
    'Starting_fine_fuel_moisture_code_max': 'max',
    'Starting_duff_moisture_code_mean': 'mean',
    'Starting_duff_moisture_code_max': 'max',
    'Starting_drought_code_mean': 'mean',
    'Starting_drought_code_max': 'max',
    'Burning_index_mean': 'mean',
    'Burning_index_max': 'max',
    'Spread_component_mean': 'mean',
    'Spread_component_max': 'max',
    'Energy_release_component_mean': 'mean',
    'Energy_release_component_max': 'max',
    'Moisture_content_herbaceous_mean': 'mean',
    'Moisture_content_herbaceous_max': 'max',
    'Moisture_content_woody_mean': 'mean',
    'Moisture_content_woody_max': 'max',
    'Dead_one_hour_fuel_moisture_mean': 'mean',
    'Dead_one_hour_fuel_moisture_max': 'max',
    'Dead_ten_hour_fuel_moisture_mean': 'mean',
    'Dead_ten_hour_fuel_moisture_max': 'max',
    'Dead_hundred_hour_fuel_moisture_mean': 'mean',
    'Dead_hundred_hour_fuel_moisture_max': 'max',
    'Dead_thousand_hour_fuel_moisture_mean': 'mean',
    'Dead_thousand_hour_fuel_moisture_max': 'max',
    'Latitude': 'first',
    'Longitude': 'first'
}

events = events.groupby(id_cols, as_index=False).agg(aggregation_rules)

events["Event_id"] = np.arange(len(events))


# ----------------------------
# 7. FINAL SAVE
# ----------------------------
logger.info("Saving...")
out_cols = [
    "Event_id", "Latitude", "Longitude", "Start_date", "End_date", "Duration", "Confidence", "Burned_area", 
    "Fire_radiative_power_mean", "Fire_radiative_power_max", 
    'Fire_weather_index_mean', 'Fire_weather_index_max', 
    'Initial_spread_index_mean', 'Initial_spread_index_max', 
    'Buildup_index_mean', 'Buildup_index_max', 
    'Starting_fine_fuel_moisture_code_mean', 'Starting_fine_fuel_moisture_code_max', 
    'Starting_duff_moisture_code_mean', 'Starting_duff_moisture_code_max', 
    'Starting_drought_code_mean', 'Starting_drought_code_max', 
    'Burning_index_mean', 'Burning_index_max', 
    'Spread_component_mean', 'Spread_component_max', 
    'Energy_release_component_mean', 'Energy_release_component_max', 
    'Moisture_content_herbaceous_mean', 'Moisture_content_herbaceous_max', 
    'Moisture_content_woody_mean', 'Moisture_content_woody_max', 
    'Dead_one_hour_fuel_moisture_mean', 'Dead_one_hour_fuel_moisture_max', 
    'Dead_ten_hour_fuel_moisture_mean', 'Dead_ten_hour_fuel_moisture_max', 
    'Dead_hundred_hour_fuel_moisture_mean', 'Dead_hundred_hour_fuel_moisture_max', 
    'Dead_thousand_hour_fuel_moisture_mean', 'Dead_thousand_hour_fuel_moisture_max',
]
events[out_cols].to_csv(output_csv, index=False)
logger.info(f"Successfully saved {len(events)} events to {output_csv}")
