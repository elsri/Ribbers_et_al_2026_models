import logging
import warnings
import cartopy.crs as ccrs
import cartopy.feature as feature
import cartopy.io.img_tiles as cimgt
import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import shapely
from cartopy.io import shapereader
from cartopy.io.shapereader import Reader
from netCDF4 import Dataset
from shapely.geometry import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

warnings.filterwarnings("ignore")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s;%(message)s")

sh.setFormatter(formatter)

logger.addHandler(sh)

def CountNonNaNs(data):
    valid = ~np.isnan(data)

    count_more_than_three = np.sum(valid & (data >= 3) & (data <=6))
    count_one_to_two = np.sum(valid & (data >= 1) & (data <= 2))
    count_matching = np.sum(valid & (data == 0))
    count_one_to_two_minus = np.sum(valid & (data <= -1) & (data >= -2))
    count_more_than_three_minus = np.sum(valid & (data <= -3) & (data >=-6))

    percentage_more_than_three = np.round(
        count_more_than_three/(count_more_than_three+count_one_to_two+count_matching+
                               count_one_to_two_minus+count_more_than_three_minus)*100,1
    )
    percentage_one_to_two = np.round(
        count_one_to_two/(count_more_than_three+count_one_to_two+count_matching+
                          count_one_to_two_minus+count_more_than_three_minus)*100,1
    )
    percentage_matching = np.round(
        count_matching/(count_more_than_three+count_one_to_two+count_matching+
                        count_one_to_two_minus+count_more_than_three_minus)*100,1
    )
    percentage_one_to_two_minus = np.round(
        count_one_to_two_minus/(count_more_than_three+count_one_to_two+count_matching+
                                count_one_to_two_minus+count_more_than_three_minus)*100,1
    )
    percentage_more_than_three_minus = np.round(
        count_more_than_three_minus/(count_more_than_three+count_one_to_two+count_matching+
                                     count_one_to_two_minus+count_more_than_three_minus)*100,1
    )
    
    return (percentage_more_than_three, percentage_one_to_two, percentage_matching,
            percentage_one_to_two_minus, percentage_more_than_three_minus)

def Plotting(fire_model, fireevent_type, dataset_peakmonth, axis1, axis2, title):
    if fire_model == "FWI" and fireevent_type == "Fire occurrence":
        ticks = [-6, -4, -2, 0, 2, 4, 6]
        colorbarlabel = "FC peak month - FWI peak month"
        barplot_title = "FC peak month -\nFWI peak month"
        dataset_peak = dataset_peakmonth["fire_occurrence_minus_fire_weather_index"][:, :]
        vmin, vcenter, vmax = (-6, 0, 6)
        names=(
            "All",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )
    elif fire_model == "NFDRS" and fireevent_type == "Fire occurrence":
        ticks = [-6, -4, -2, 0, 2, 4, 6]
        colorbarlabel = "FC peak month - BI peak month"
        barplot_title = "FC peak month -\nBI peak month"
        dataset_peak = dataset_peakmonth["fire_occurrence_minus_burning_index"][:, :]
        vmin, vcenter, vmax = (-6, 0, 6)
        names=(
            "All",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )
    elif fire_model == "FWI" and fireevent_type == "FRP":
        ticks = [-6, -4, -2, 0, 2, 4, 6]
        colorbarlabel = "FRP peak month - FWI peak month"
        barplot_title = "FRP peak month -\nFWI peak month"
        dataset_peak = dataset_peakmonth["frp_minus_fire_weather_index"][:,:]
        vmin, vcenter, vmax = (-6, 0, 6)
        names=(
            "All",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )
    elif fire_model == "NFDRS" and fireevent_type == "FRP":
        ticks = [-6, -4, -2, 0, 2, 4, 6]
        colorbarlabel = "FRP peak month - BI peak month"
        barplot_title = "FRP peak month -\nBI peak month"
        dataset_peak = dataset_peakmonth["frp_minus_burning_index"][:,:]
        vmin, vcenter, vmax = (-6, 0, 6)
        names=(
            "All",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )

    landcover = dataset_peakmonth["landcover"][:, :]
    coniferous_data = np.where(landcover == 19, dataset_peak, np.nan)
    deciduous_data = np.where(landcover == 20, dataset_peak, np.nan)
    mixed_data = np.where(landcover == 21, dataset_peak, np.nan)
    
    (percentage_more_than_three, percentage_one_to_two, percentage_matching,
     percentage_one_to_two_minus, percentage_more_than_three_minus) = CountNonNaNs(data = dataset_peak)
    #print(f"Percentage of all {fire_model} {fireevent_type} gridcells with month lag > 3: {percentage_more_than_three}")
    #print(f"Percentage of all {fire_model} {fireevent_type} gridcells with month lag > 1: {percentage_one_to_two}")
    #print(f"Percentage of all {fire_model} {fireevent_type} gridcells with matching peak month: {percentage_matching}")
    #print(f"Percentage of all {fire_model} {fireevent_type} gridcells with month lag < -1: {percentage_one_to_two_minus}")
    #print(f"Percentage of all {fire_model} {fireevent_type} gridcells with month lag < -3: {percentage_more_than_three_minus}")
    #print(f"Percentage of all {fire_model} {fireevent_type} gridcells with index preceding fire peak with more than one month: {percentage_more_than_three_minus + percentage_one_to_two_minus}")
    #print(f"Percentage of all {fire_model} {fireevent_type} gridcells with index following fire peak with more than one month: {percentage_one_to_two + percentage_more_than_three}")
    
    (percentage_coniferous_more_than_three, percentage_coniferous_one_to_two, percentage_coniferous_matching,
     percentage_coniferous_one_to_two_minus, percentage_coniferous_more_than_three_minus) = CountNonNaNs(data = coniferous_data)
    #print(f"Percentage of coniferous {fire_model} {fireevent_type} gridcells with month lag > 3: {percentage_coniferous_more_than_three}")
    #print(f"Percentage of coniferous {fire_model} {fireevent_type} gridcells with month lag > 1: {percentage_coniferous_one_to_two}")
    #print(f"Percentage of coniferous {fire_model} {fireevent_type} gridcells with matching peak month: {percentage_coniferous_matching}")
    #print(f"Percentage of coniferous {fire_model} {fireevent_type} gridcells with month lag < -1: {percentage_coniferous_one_to_two_minus}")
    #print(f"Percentage of coniferous {fire_model} {fireevent_type} gridcells with month lag < -3: {percentage_coniferous_more_than_three_minus}")
    #print(f"Percentage of coniferous {fire_model} {fireevent_type} gridcells with index preceding fire peak with more than one month: {percentage_coniferous_more_than_three_minus + percentage_coniferous_one_to_two_minus}")
    #print(f"Percentage of coniferous {fire_model} {fireevent_type} gridcells with index following fire peak with more than one month: {percentage_coniferous_one_to_two + percentage_coniferous_more_than_three}")
    
    (percentage_deciduous_more_than_three, percentage_deciduous_one_to_two, percentage_deciduous_matching,
     percentage_deciduous_one_to_two_minus, percentage_deciduous_more_than_three_minus) = CountNonNaNs(data = deciduous_data)
    #print(f"Percentage of deciduous {fire_model} {fireevent_type} gridcells with month lag > 3: {percentage_deciduous_more_than_three}")
    #print(f"Percentage of deciduous {fire_model} {fireevent_type} gridcells with month lag > 1: {percentage_deciduous_one_to_two}")
    #print(f"Percentage of deciduous {fire_model} {fireevent_type} gridcells with matching peak month: {percentage_deciduous_matching}")
    #print(f"Percentage of deciduous {fire_model} {fireevent_type} gridcells with month lag < -1: {percentage_deciduous_one_to_two_minus}")
    #print(f"Percentage of deciduous {fire_model} {fireevent_type} gridcells with month lag < -3: {percentage_deciduous_more_than_three_minus}")
    #print(f"Percentage of deciduous {fire_model} {fireevent_type} gridcells with index preceding fire peak with more than one month: {percentage_deciduous_more_than_three_minus + percentage_deciduous_one_to_two_minus}")
    #print(f"Percentage of deciduous {fire_model} {fireevent_type} gridcells with index following fire peak with more than one month: {percentage_deciduous_one_to_two + percentage_deciduous_more_than_three}")
    
    (percentage_mixed_more_than_three, percentage_mixed_one_to_two, percentage_mixed_matching,
     percentage_mixed_one_to_two_minus, percentage_mixed_more_than_three_minus) = CountNonNaNs(data = mixed_data)
    #print(f"Percentage of mixed {fire_model} {fireevent_type} gridcells with month lag > 3: {percentage_mixed_more_than_three}")
    #print(f"Percentage of mixed {fire_model} {fireevent_type} gridcells with month lag > 1: {percentage_mixed_one_to_two}")
    #print(f"Percentage of mixed {fire_model} {fireevent_type} gridcells with matching peak month: {percentage_mixed_matching}")
    #print(f"Percentage of mixed {fire_model} {fireevent_type} gridcells with month lag < -1: {percentage_mixed_one_to_two_minus}")
    #print(f"Percentage of mixed {fire_model} {fireevent_type} gridcells with month lag < -3: {percentage_mixed_more_than_three_minus}")
    #print(f"Percentage of mixed {fire_model} {fireevent_type} gridcells with index preceding fire peak with more than one month: {percentage_mixed_more_than_three_minus + percentage_mixed_one_to_two_minus}")
    #print(f"Percentage of mixed {fire_model} {fireevent_type} gridcells with index following fire peak with more than one month: {percentage_mixed_one_to_two + percentage_mixed_more_than_three}")
    
    value_name_more_than_three = "+3 to 6 months"
    value_name_one_to_two = "+1 to 2 months"
    value_name_matching = "Matching"
    value_name_one_to_two_minus = "-1 to 2 months"
    value_name_more_than_three_minus = "-3 to 6 months"

    '''
    norm = TwoSlopeNorm(
        vmin = vmin,
        vcenter = vcenter,
        vmax = vmax
    )
    
    levels = np.concatenate([
        np.linspace(vmin, vcenter, 20),
        np.linspace(vcenter, vmax, 20)[1:]
    ])
    '''
    positive_value = axs[axis1, axis2].contourf(
        lons,
        lats,
        dataset_peak,
        transform=transform,
        zorder=zorder_fire_index,
        cmap=category_cmap,
        levels=bounds,
        norm=norm,
        extend="both",
    )

    axs[axis1, axis2].add_feature(
        feature.BORDERS, linewidth=linewidth_countrylines, zorder=zorder_countrylines
    )
    axs[axis1, axis2].set_extent(extent)
    axs[axis1, axis2].add_geometries(
        shapely.get_parts(poly[0]),
        crs=transform,
        facecolor=facecolor_polygon,
        edgecolor=edgecolor_polygon,
        linewidth=linewidth_polygon,
        zorder=zorder_polygon,
    )
    axs[axis1, axis2].add_geometries(
        msk_stm,
        st_proj,
        facecolor=facecolor_landmask,
        edgecolor=edgecolor_landmask,
        zorder=zorder_landmask,
    )
    axs[axis1, axis2].axis("off")
    axs[axis1, axis2].set_title(title, loc="left", zorder = zorder_title, fontsize=labelsize+5)

    '''
    cbar = fig.colorbar(
        positive_value,
        ax=axs[axis1, axis2],
        ticks=ticks,
        orientation="vertical",
        shrink=1,
        pad=0.01,
    )
    cbar.set_label(colorbarlabel, size=fontsize, labelpad=15)
    cbar.ax.tick_params(labelsize=labelsize)
    '''
    
    # BARPLOT
    values = {
        value_name_more_than_three_minus: np.array([
            percentage_more_than_three_minus,
            percentage_coniferous_more_than_three_minus,
            percentage_deciduous_more_than_three_minus,
            percentage_mixed_more_than_three_minus
        ]),
        value_name_one_to_two_minus: np.array([
            percentage_one_to_two_minus,
            percentage_coniferous_one_to_two_minus,
            percentage_deciduous_one_to_two_minus,
            percentage_mixed_one_to_two_minus
        ]),
        value_name_matching: np.array([
            percentage_matching,
            percentage_coniferous_matching,
            percentage_deciduous_matching,
            percentage_mixed_matching
        ]),
        value_name_one_to_two: np.array([
            percentage_one_to_two,
            percentage_coniferous_one_to_two,
            percentage_deciduous_one_to_two,
            percentage_mixed_one_to_two
        ]),
        value_name_more_than_three: np.array([
            percentage_more_than_three,
            percentage_coniferous_more_than_three,
            percentage_deciduous_more_than_three,
            percentage_mixed_more_than_three
        ]),
    }

    colors = {
        value_name_more_than_three: colour_more_than_three,
        value_name_one_to_two: colour_one_to_two,
        value_name_matching: colour_matching,
        value_name_one_to_two_minus: colour_one_to_two_minus,
        value_name_more_than_three_minus: colour_more_than_three_minus
    }

    width = 0.5
    left = np.zeros(len(names))
    
    # Inset axex
    axins = inset_axes(
        axs[axis1, axis2],
        width="20%",      # relative to map width
        height="15%",     # relative to map height
        loc="upper left",
        bbox_to_anchor=(0.1, -0.08, 1, 1),
        bbox_transform=axs[axis1, axis2].transAxes,
        borderpad=0
    )

    for label, value in values.items():
        axins.bar(
            names,
            value,
            bottom=left,
            color=colors[label],
        )
        left += value
    axins.set_ylim(0,100)

    axins.tick_params(
        axis='both',
        labelsize=labelsize
    )
    axins.tick_params(
        axis='x',
        rotation = 90
    )

    axins.set_title(
        barplot_title, fontsize = fontsize, pad=25
    )

    axins.yaxis.set_major_formatter(mtick.PercentFormatter())

    axins.set_facecolor("white")
    axins.patch.set_alpha(0.9)

    for spine in axins.spines.values():
        spine.set_linewidth(1)

    return (
        axs,
    )

def rect_from_bound(xmin, xmax, ymin, ymax):
    # Returns list of (x,y)'s for a rectangle
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]

logger.info(" Setting the plotting parameters...")

# Plots
nrows = 2
ncolumns = 2

# Map parameters
projection = ccrs.Mercator()
transform = ccrs.PlateCarree()
extent = [4.5, 31.9, 54.7, 71.24]
central_lon, central_lat = 0, 45

# Colourmaps
colors = ["#0329ffff", "#0329ffff", "#0fffccff","#4ce15eff","#cbff73ff", "#dbcd00ff", "#dbcd00ff"]
colourmap_fire = LinearSegmentedColormap.from_list("mycmap", colors)

#colourmap_fire = "berlin"
#colour_more_than_three_minus = "#192d99ff"
colour_more_than_three_minus = "blue"
#colour_one_to_two_minus = "#32887aff"
colour_one_to_two_minus = "red"
#colour_matching = "#4ce15eff"
colour_matching = "white"
#colour_one_to_two = "#a9e83fff"
colour_one_to_two = "lightpink"
#colour_more_than_three = "#fff024ff"
colour_more_than_three = "lightblue"
colourmap_fire_binary = matplotlib.colors.ListedColormap(['darkturquoise', 'salmon'])
colourmap_nofire_binary = matplotlib.colors.ListedColormap(['salmon', 'darkturquoise'])
colourmap_forest_hclim3 = matplotlib.colors.LinearSegmentedColormap.from_list(
    "my_colormap", ("white", "white"), N=2, gamma=1.0
)

category_cmap = ListedColormap([
    colour_more_than_three_minus,  # -6 to -3
    colour_one_to_two_minus,       # -2 to -1
    colour_matching,               # 0
    colour_one_to_two,             # 1 to 2
    colour_more_than_three         # 3 to 6
])
bounds = [-6.5, -2.5, -0.5, 0.5, 2.5, 6.5]

norm = BoundaryNorm(bounds, category_cmap.N)

# Colours
color_damage = "lightgreen"
facecolor_landmask = "#f3f8ff"
facecolor_polygon = "#808080ff"
edgecolor_landmask = "none"
edgecolor_polygon = "black"
facecolor_oceanmask = "#f3f8ff"
edgecolor_oceanmask = "none"
transparency_damage = 0.75
transparency_ocean = 1

fontsize = 45
labelsize = 40

# Zorders
zorder_landmask = 3
zorder_polygon = 1
zorder_countrylines = 2
zorder_fire_index = 4
zorder_legend = 5
zorder_title = 6

# Linewidths
linewidth_polygon = 2.5
linewidth_countrylines = 2.5
linewidth_municipality = 1

logger.info(" Setting the map boundaries...")

# Get borders
resolution = "10m"
category = "cultural"
name = "admin_0_countries"

# Read the shapefile using geopandas
shapefile_name_countries = shapereader.natural_earth(resolution, category, name)
shapefile_name_municipalities = "/home/els/Nextcloud/Documents/PhD/damage_index/DamageData/NorwayShapefile/NorwayKommuner.shp"

shapefile_countries = geopandas.read_file(shapefile_name_countries)
shapefile_municipalities = geopandas.read_file(shapefile_name_municipalities)

reader_municipalities = Reader(shapefile_name_municipalities)
municipalities = reader_municipalities.records()

# Load fire data
geometries_municipalities = []
for municipality in municipalities:
    geometries_municipalities.append(municipality.geometry)

# Select countries that need to be visible
scan3 = shapefile_countries[shapefile_countries["ADMIN"].isin(["Norway", "Sweden", "Finland"])]
scan3_dissolved = scan3.dissolve(by="LEVEL")
poly = [scan3_dissolved["geometry"].values[0]]

# Create the mask
stamen_terrain = cimgt.Stamen("terrain_background")
st_proj = stamen_terrain.crs  # projection used by Stamen images
ll_proj = ccrs.PlateCarree()

pad1 = 0.1  # padding, degrees unit
exts = [
    poly[0].bounds[0] - pad1,
    poly[0].bounds[2] + pad1,
    poly[0].bounds[1] - pad1,
    poly[0].bounds[3] + pad1,
]
msk = Polygon(rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
msk_stm = st_proj.project_geometry(
    msk, ll_proj
)  # project geometry to the projection used by stamen

# Coords definition

#---------------------------------------------------------------------------
logger.info(" Calculating percentages...")
dataset_peakmonth = Dataset(
    "/PATH/TO/PeakFireMonths_BI_FWI_FRP_HCLIM3_2001_2018.nc",
    "r",
)

lats = dataset_peakmonth["lat"][:, :]
lons = dataset_peakmonth["lon"][:, :]

#---------------------------------------------------------------------------
'''
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncolumns,
    figsize=(40,45),
    subplot_kw={"projection": projection},
    layout="constrained"
)
'''


fig = plt.figure(figsize=(35, 50), layout="constrained")

gs = GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[0.10, 1, 1]
)

ax_leg = fig.add_subplot(gs[0, :])
ax_leg.axis("off")

axs = np.empty((2,2), dtype=object)
axs[0,0] = fig.add_subplot(gs[1,0], projection=projection)
axs[0,1] = fig.add_subplot(gs[1,1], projection=projection)
axs[1,0] = fig.add_subplot(gs[2,0], projection=projection)
axs[1,1] = fig.add_subplot(gs[2,1], projection=projection)


logger.info(" Setting the data boundaries...")


min_values = 0
max_values = 1
levels_peakmonth = np.linspace(min_values, max_values, abs(max_values) + 1)

plt.subplots_adjust(top=0.85)

logger.info(" Plotting the data...")

'''
FWI90 Fire map
'''
axs[0,0] = Plotting(
    fire_model = "FWI",
    fireevent_type = "Fire occurrence",
    dataset_peakmonth = dataset_peakmonth,
    axis1 = 0,
    axis2 = 0,
    title = "a."
)

'''
FWI90 FRP map
'''
axs[0,1] = Plotting(
    fire_model = "FWI",
    fireevent_type = "FRP",
    dataset_peakmonth = dataset_peakmonth,
    axis1 = 0,
    axis2 = 1,
    title = "b."
)

'''
NFDRS90 Fire map
'''
axs[1,0] = Plotting(
    fire_model = "NFDRS",
    fireevent_type = "Fire occurrence",
    dataset_peakmonth = dataset_peakmonth,
    axis1 = 1,
    axis2 = 0,
    title = "c."
)

'''
NFDRS90 FRP map
'''
axs[1,1] = Plotting(
    fire_model = "NFDRS",
    fireevent_type = "FRP",
    dataset_peakmonth = dataset_peakmonth,
    axis1 = 1,
    axis2 = 1,
    title = "d."
)



legend_elements = [
    Patch(facecolor=colour_more_than_three_minus, edgecolor="black", label="-3 to 6 months"),
    Patch(facecolor=colour_one_to_two_minus, edgecolor="black", label="-1 to 2 months"),
    Patch(facecolor=colour_matching, edgecolor="black", label="0 months"),
    Patch(facecolor=colour_one_to_two, edgecolor="black", label="+1 to 2 months"),
    Patch(facecolor=colour_more_than_three, edgecolor="black", label="+3 to 6 months"),
]

ax_leg.legend(
    handles=legend_elements,
    loc="center",
    ncol=5,
    fontsize=labelsize,
    title="Peak month offset",
    title_fontsize = fontsize,
    #frameon=True,
    #bbox_to_anchor=(0.5, 0.99)
)

logger.info(" Saving figure...")

plt.savefig(
    "/PATH/TO/peakmonth_HCLIM3.png"
)

