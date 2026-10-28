import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(message)s")
logger = logging.getLogger(__name__)

def build_spatial_map(ds, var_name):
    # We only build the tree from valid (non-NaN) land cells
    # This ensures that even if a fire is on the coast, we find the nearest LAND
    mask = ~np.isnan(ds[var_name].isel(time=0).values)
    lat_f = ds["lat"].values[mask]
    lon_f = ds["lon"].values[mask]
    # Keep track of the original 2D indices of these land cells
    y_coords, x_coords = np.where(mask)
    tree = cKDTree(np.column_stack([lat_f, lon_f]))
    return tree, y_coords, x_coords

def get_stats_for_event(ds, tree, y_orig, x_orig, var_name, lat, lon, date, radius):
    t_start = date
    t_end = date + pd.Timedelta(days=5)
    
    # 1. Slice time first
    time_slice = ds[var_name].sel(time=slice(t_start, t_end))
    if time_slice.time.size == 0:
        return np.nan, np.nan

    # 2. Find indices within radius
    indices = tree.query_ball_point([lat, lon], radius)
    
    # Fallback: If radius is empty or all NaN, get the 1 absolute nearest land neighbor
    if not indices:
        _, dist_idx = tree.query([lat, lon], k=1)
        indices = [dist_idx]

    # Map the tree indices back to the original y, x coordinates
    y_idxs = y_orig[indices]
    x_idxs = x_orig[indices]
    
    # 3. Extract and calculate
    subset = time_slice.isel(y=xr.DataArray(y_idxs, dims="pts"), 
                             x=xr.DataArray(x_idxs, dims="pts"))
    
    return float(subset.mean(skipna=True)), float(subset.max(skipna=True))

# ---------------------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------------------
input_csv = "/PATH/TO/dsb_burned_area_events.csv"
fwi_file = "/PATH/TO/FWI_HCLIM3_Fennoscandia_2001_2018.nc"
bi_file = "/PATH/TO/NFDR_HCLIM3_Fennoscandia_2001_2018.nc"
output_csv = "/PATH/TO/fire_events_dsb.csv"

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv(input_csv, sep=';')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
logger.info(f"Loaded {len(df)} events. Processing indices...")

fwi_ds = xr.open_dataset(fwi_file)
bi_ds = xr.open_dataset(bi_file)

logger.info("Building land-only spatial trees (this may take a minute)...")
fwi_tree, fwi_y, fwi_x = build_spatial_map(fwi_ds, 'fire_weather_index')
bi_tree, bi_y, bi_x = build_spatial_map(bi_ds, 'burning_index')

# ----------------------------
# PROCESSING LOOP
# ----------------------------
final_data = []

for idx, row in df.iterrows():
    lat, lon = row['Latitude'], row['Longitude']
    area = row['Total_burned_area']
    
    # Dynamic radius (min 0.05 degrees)
    radius = max((np.sqrt((area * 1000) / np.pi) / 111320.0), 0.05) # Conversion ha -> m2. Radius based on total burned area.
    
    f_mean, f_max = get_stats_for_event(fwi_ds, fwi_tree, fwi_y, fwi_x, 
                                        'fire_weather_index', lat, lon, row['Date'], radius)
    
    fs_mean, fs_max = get_stats_for_event(fwi_ds, fwi_tree, fwi_y, fwi_x, 
                                        'initial_spread_index', lat, lon, row['Date'], radius)
    
    b_mean, b_max = get_stats_for_event(bi_ds, bi_tree, bi_y, bi_x, 
                                        'burning_index', lat, lon, row['Date'], radius)
    bs_mean, bs_max = get_stats_for_event(bi_ds, bi_tree, bi_y, bi_x, 
                                        'spread_component', lat, lon, row['Date'], radius)

    final_data.append({
        "Latitude": lat, "Longitude": lon, "Date": row['Date'].strftime('%Y-%m-%d'),
        "Burned area": area, 
        "Mean fire weather index": f_mean, "Max fire weather index": f_max, 
        "Mean burning index": b_mean, "Max burning index": b_max,
        "Mean initial spread index": fs_mean, "Max initial spread index": fs_max,
        "Mean spread component": bs_mean, "Max spread component": bs_max,
    })

# ----------------------------
# SAVE OUTPUT
# ----------------------------
logger.info("Saving output...")

out_df = pd.DataFrame(final_data)
out_df.to_csv(output_csv, index=False, sep=';')

logger.info(f"Success! Output saved to: {output_csv}")
