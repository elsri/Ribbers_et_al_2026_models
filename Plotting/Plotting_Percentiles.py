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

    count_positive = np.sum(valid & (data > 1))
    count_neutral = np.sum(valid & (data == 1))
    #count_negative = np.sum(valid & (data < 1) & (data > 0))
    count_negative = np.sum(valid & (data < 1))

    percentage_positive = np.round(
        count_positive/(count_positive+count_neutral+count_negative)*100,1
    )
    percentage_neutral = np.round(
        count_neutral/(count_positive+count_neutral+count_negative)*100,1
    )
    percentage_negative = np.round(
        count_negative/(count_positive+count_neutral+count_negative)*100,1
    )
    
    return (percentage_positive, percentage_neutral, percentage_negative)

def Plotting(fire_model, fireevent_type, dataset_percentiles, axis1, axis2, title):
    if fire_model == "FWI" and fireevent_type == "Fire occurrence":
        ticks = [0, 0.25, 0.5, 0.75, 1, 3.25, 5.5, 7.75, 10]
        colorbarlabel = "Probability of FWI > FWI90 leading to FC > 0"
        barplot_title = "P(FC > 0 | high FWI)\n--------------------------\nP(FC > 0)"
        dataset_high = dataset_percentiles["prob_fire_given_fwi90"][:, :]
        dataset_low = dataset_percentiles["prob_fire_given_fwi10"][:, :]
        vmin, vcenter, vmax = (0, 1, 10)
        names=(
            "FWI90",
            "FWI10",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )
    elif fire_model == "NFDRS" and fireevent_type == "Fire occurrence":
        ticks = [0, 0.25, 0.5, 0.75, 1, 3.25, 5.5, 7.75, 10]
        colorbarlabel = "Probability of BI > BI90 leading to FC > 0"
        barplot_title = "P(FC > 0 | high BI)\n--------------------------\nP(FC > 0)"
        dataset_high = dataset_percentiles["prob_fire_given_nfdrs90"][:, :]
        dataset_low = dataset_percentiles["prob_fire_given_nfdrs10"][:, :]
        vmin, vcenter, vmax = (0, 1, 10)
        names=(
            "BI90",
            "BI10",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )
    elif fire_model == "FWI" and fireevent_type == "FRP":
        ticks = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        colorbarlabel = "Probability of FWI > FWI90 leading to FRP > FRP90"
        barplot_title = "P(high FRP | high FWI)\n--------------------------\nP(high FRP)"
        dataset_high = dataset_percentiles["prob_high_frp_given_fwi90"][:,:]
        dataset_low = dataset_percentiles["prob_high_frp_given_fwi10"][:,:]
        vmin, vcenter, vmax = (0, 1, 2)
        names=(
            "FWI90",
            "FWI10",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )
    elif fire_model == "NFDRS" and fireevent_type == "FRP":
        ticks = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        colorbarlabel = "Probability of BI > BI90 leading to FRP > FRP90"
        barplot_title = "P(high FRP | high BI)\n--------------------------\nP(high FRP)"
        dataset_high = dataset_percentiles["prob_high_frp_given_nfdrs90"][:,:]
        dataset_low = dataset_percentiles["prob_high_frp_given_nfdrs10"][:,:]
        vmin, vcenter, vmax = (0, 1, 2)
        names=(
            "BI90",
            "BI10",
            "Coniferous",
            "Deciduous",
            "Mixed"
        )

    landcover = dataset_percentiles["landcover"][:, :]
    coniferous_data = np.where(landcover == 19, dataset_high, np.nan)
    deciduous_data = np.where(landcover == 20, dataset_high, np.nan)
    mixed_data = np.where(landcover == 21, dataset_high, np.nan)
    
    percentage_high_positive, percentage_high_neutral, percentage_high_negative = CountNonNaNs(data = dataset_high)
    #print(f"Percentage of all high {fire_model} {fireevent_type} gridcells with ratio > 1: {percentage_high_positive}")
    #print(f"Percentage of all high {fire_model} {fireevent_type} gridcells with ratio = 1: {percentage_high_neutral}")
    #print(f"Percentage of all high {fire_model} {fireevent_type} gridcells with ratio < 1: {percentage_high_negative}")
    percentage_low_positive, percentage_low_neutral, percentage_low_negative = CountNonNaNs(data = dataset_low)
    #print(f"Percentage of all low {fire_model} {fireevent_type} gridcells with ratio > 1: {percentage_low_positive}")
    #print(f"Percentage of all low {fire_model} {fireevent_type} gridcells with ratio = 1: {percentage_low_neutral}")
    #print(f"Percentage of all low {fire_model} {fireevent_type} gridcells with ratio < 1: {percentage_low_negative}")
    percentage_coniferous_positive, percentage_coniferous_neutral, percentage_coniferous_negative = CountNonNaNs(data = coniferous_data)
    #print(f"Percentage of all coniferous {fire_model} {fireevent_type} gridcells with ratio > 1: {percentage_coniferous_positive}")
    #print(f"Percentage of all coniferous {fire_model} {fireevent_type} gridcells with ratio = 1: {percentage_coniferous_neutral}")
    #print(f"Percentage of all coniferous {fire_model} {fireevent_type} gridcells with ratio < 1: {percentage_coniferous_negative}")
    percentage_deciduous_positive, percentage_deciduous_neutral, percentage_deciduous_negative = CountNonNaNs(data = deciduous_data)
    #print(f"Percentage of all deciduous {fire_model} {fireevent_type} gridcells with ratio > 1: {percentage_deciduous_positive}")
    #print(f"Percentage of all deciduous {fire_model} {fireevent_type} gridcells with ratio = 1: {percentage_deciduous_neutral}")
    #print(f"Percentage of all deciduous {fire_model} {fireevent_type} gridcells with ratio < 1: {percentage_deciduous_negative}")
    percentage_mixed_positive, percentage_mixed_neutral, percentage_mixed_negative = CountNonNaNs(data = mixed_data)
    #print(f"Percentage of all mixed {fire_model} {fireevent_type} gridcells with ratio > 1: {percentage_mixed_positive}")
    #print(f"Percentage of all mixed {fire_model} {fireevent_type} gridcells with ratio = 1: {percentage_mixed_neutral}")
    #print(f"Percentage of all mixed {fire_model} {fireevent_type} gridcells with ratio < 1: {percentage_mixed_negative}")

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
        dataset_high,
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
    
    legend_elements = [
        Patch(facecolor=colour_positive, edgecolor="black", label=value_name_positive),
        Patch(facecolor=colour_neutral, edgecolor="black", label=value_name_neutral),     
        Patch(facecolor=colour_negative, edgecolor="black", label=value_name_negative),
    ]

    leg = axs[axis1, axis2].legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=labelsize,
        framealpha=0.9,
        facecolor="white",
    )
    leg.set_zorder(zorder_legend)
    '''
    # BARPLOT
    values = {
        value_name_negative: np.array([
            percentage_high_negative,
            percentage_low_negative,
            percentage_coniferous_negative,
            percentage_deciduous_negative,
            percentage_mixed_negative
        ]),
        value_name_neutral: np.array([
            percentage_high_neutral,
            percentage_low_neutral,
            percentage_coniferous_neutral,
            percentage_deciduous_neutral,
            percentage_mixed_neutral
        ]),
        value_name_positive: np.array([
            percentage_high_positive,
            percentage_low_positive,
            percentage_coniferous_positive,
            percentage_deciduous_positive,
            percentage_mixed_positive
        ]),
    }

    colors = {
        value_name_positive: colour_positive,
        value_name_neutral: colour_neutral,
        value_name_negative: colour_negative
    }

    width = 0.5
    left = np.zeros(len(names))
    
    # Inset axex
    axins = inset_axes(
        axs[axis1, axis2],
        width="20%",      # relative to map width
        height="15%",     # relative to map height
        loc="upper left",
        bbox_to_anchor=(0.1, -0.12, 1, 1),
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
        barplot_title, fontsize = fontsize, pad = 20
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
colors = ["#0329ffff", "#0fffccff","#4ce15eff","#cbff73ff", "#dbcd00ff"]
colourmap_fire = LinearSegmentedColormap.from_list("mycmap", colors)

#colourmap_fire = "berlin"
#colour_negative = "#192d99ff"
colour_negative = "blue"
#colour_neutral = "#4ce15eff"
colour_neutral = "white"
#colour_positive = "#fff024ff"
colour_positive = "red"

category_cmap = ListedColormap([
    colour_negative,  # ratio < 1
    colour_neutral,   # ratio = 1
    colour_positive,  # ratio > 1 
])
bounds = [-10.5, 0.5, 1.5, 10.5]

norm = BoundaryNorm(bounds, category_cmap.N)

#colourmap_fire_binary = matplotlib.colors.ListedColormap(['darkturquoise', 'salmon'])
#colourmap_nofire_binary = matplotlib.colors.ListedColormap(['salmon', 'darkturquoise'])
#colourmap_forest_hclim3 = matplotlib.colors.LinearSegmentedColormap.from_list(
#    "my_colormap", ("white", "white"), N=2, gamma=1.0
#)

# Colours
color_damage = "lightgreen"
facecolor_landmask = "#f3f8ff"
facecolor_polygon = "#808080ff"
facecolor_polygon = "none"
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

value_name_positive = "Ratio > 1"
value_name_neutral = "Ratio = 1"
value_name_negative = "Ratio < 1"

logger.info(" Setting the map boundaries...")

# Get borders
resolution = "10m"
category = "cultural"
name = "admin_0_countries"

# Read the shapefile using geopandas
shapefile_name_countries = shapereader.natural_earth(resolution, category, name)
shapefile_name_municipalities = "/PATH/TO/NorwayKommuner.shp"

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
dataset_percentiles = Dataset(
    "/PATH/TO/percentiles_hclim3.nc",
    "r",
)
lats = dataset_percentiles["lat"][:, :]
lons = dataset_percentiles["lon"][:, :]

#---------------------------------------------------------------------------
'''
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=ncolumns,
    figsize=(40,45),
    subplot_kw={"projection": projection},
    layout="tight"
)

logger.info(" Setting the data boundaries...")


min_values = 0
max_values = 1
levels_percentiles = np.linspace(min_values, max_values, abs(max_values) + 1)
'''
fig = plt.figure(figsize=(35, 50), layout="constrained")

gs = GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[0.05, 1, 1]
)

ax_leg = fig.add_subplot(gs[0, :])
ax_leg.axis("off")

axs = np.empty((2,2), dtype=object)
axs[0,0] = fig.add_subplot(gs[1,0], projection=projection)
axs[0,1] = fig.add_subplot(gs[1,1], projection=projection)
axs[1,0] = fig.add_subplot(gs[2,0], projection=projection)
axs[1,1] = fig.add_subplot(gs[2,1], projection=projection)

logger.info(" Plotting the data...")

'''
FWI90 Fire map
'''
axs[0,0] = Plotting(
    fire_model = "FWI",
    fireevent_type = "Fire occurrence",
    dataset_percentiles = dataset_percentiles,
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
    dataset_percentiles = dataset_percentiles,
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
    dataset_percentiles = dataset_percentiles,
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
    dataset_percentiles = dataset_percentiles,
    axis1 = 1,
    axis2 = 1,
    title = "d."
)

legend_elements = [
    Patch(facecolor=colour_positive, edgecolor="black", label=value_name_positive),
    Patch(facecolor=colour_neutral, edgecolor="black", label=value_name_neutral),     
    Patch(facecolor=colour_negative, edgecolor="black", label=value_name_negative),
]

ax_leg.legend(
    handles=legend_elements,
    loc="center",
    ncol=5,
    fontsize=labelsize,
    #framealpha=0.9,
    #facecolor="white",
)

logger.info(" Saving figure...")

plt.savefig(
    "/PATH/TO/percentiles_HCLIM3.png"
)

