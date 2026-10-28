import logging
import warnings
import numpy as np
import pandas
import shapely
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap, LinearSegmentedColormap
#from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def get_log_linear_stats(burned_area_range, x_values, y_values):
    """Returns smooth y-values and the Pearson R2"""
    mask = (x_values > 0) & (y_values > 0)

    x_filtered = x_values[mask]
    y_filtered = y_values[mask]

    x_log = np.log10(x_filtered)
    y_log = np.log10(y_filtered)
    """Returns smooth y-values and the Pearson R2"""
    # Fit
    coeffs = np.polyfit(x_log, y_log, 1)
    poly_func = np.poly1d(coeffs)

    y_log_pred = poly_func(x_log)
    residuals = y_log - y_log_pred
    
    # Pearson R2 (The 'Calc' style)
    #r_squared = np.corrcoef(x_log, y_log)[0, 1]**2
    r, p_value = pearsonr(x_log, y_log)
    r_squared = r**2
    
    # Smooth line for plotting
    y_smooth = 10 ** (poly_func(np.log10(burned_area_range)))
    
    return y_smooth, r_squared, p_value, residuals, x_log

def format_decimals(value):
    if value >= 0.001:
        return f"{value:.3f}"
    else:
        return f"{value:.2e}"

def plottings(x_data1, x_data2, y_data1, y_data2, dataset1_name, dataset2_name, index_x,
              index_y, min_y, max_y, stepsize, plot_number, legend_label1, legend_label2,
              x_label, y_label):
    sort_dataset1 = np.argsort(x_data1)
    sort_dataset2 = np.argsort(x_data2)
    dataset1_x_sorted = np.array(x_data1)[sort_dataset1]
    dataset2_x_sorted = np.array(x_data2)[sort_dataset2]
    
    dataset1_range = np.logspace(np.log10(dataset1_x_sorted.min()),
                                 np.log10(dataset1_x_sorted.max()),
                                 200)
    dataset2_range = np.logspace(np.log10(dataset2_x_sorted.min()),
                                 np.log10(dataset2_x_sorted.max()),
                                 200)
    dataset1_y_sorted = np.array(y_data1)[sort_dataset1]
    dataset2_y_sorted = np.array(y_data2)[sort_dataset2]

    dataset1_smooth, r_squared_dataset1, p_value1, residuals1, x_log1 = get_log_linear_stats(dataset1_range,
                                                               dataset1_x_sorted,
                                                               dataset1_y_sorted)
    dataset2_smooth, r_squared_dataset2, p_value2, residuals2, x_log2 = get_log_linear_stats(dataset2_range,
                                                               dataset2_x_sorted,
                                                               dataset2_y_sorted)

    scatter_factor1 = 10 ** np.std(residuals1)
    scatter_factor2 = 10 ** np.std(residuals2)

    plot1 = axs[index_x, index_y].scatter(
        x=x_data1,
        y=y_data1,
        c="salmon",
        s=10,
        marker="v",
        alpha=0.2,
        label=f"{dataset1_name} (n={len(y_data1)})",
    )

    plot2 = axs[index_x, index_y].scatter(
        x=x_data2,
        y=y_data2,
        c="deepskyblue",
        s=10,
        marker="o",
        alpha=0.2,
        label=f"{dataset2_name} (n={len(y_data2)})",
    )

    axs[index_x, index_y].plot(dataset1_range, dataset1_smooth, "r-",
                               label=f"Power trendline ({dataset1_name})\nR²={format_decimals(r_squared_dataset1)}, p={format_decimals(p_value1)}, scatter≈x{scatter_factor1:.2f}")
    axs[index_x, index_y].plot(dataset2_range, dataset2_smooth, "b--",
                               label=f"Power trendline ({dataset2_name})\nR²={format_decimals(r_squared_dataset2)}, p={format_decimals(p_value2)}, scatter≈x{scatter_factor2:.2f}")    
    
    axs[index_x, index_y].set_xscale("log")
    axs[index_x, index_y].set_ylim([min_y,max_y])
    axs[index_x, index_y].yaxis.set_ticks(np.arange(0,max_y+1,stepsize))
    axs[index_x, index_y].legend(loc=0)
    axs[index_x, index_y].set_xlabel(x_label, size=fontsize)
    axs[index_x, index_y].set_ylabel(y_label, size=fontsize)
    axs[index_x, index_y].set_title(plot_number, loc="left")

    return axs

####################################################################
# --------------- Logging and warning settings ---------------
warnings.filterwarnings("ignore")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s;%(message)s")

sh.setFormatter(formatter)

logger.addHandler(sh)

logger.info("Loading the data...")
####################################################################
# --------------- Plot settings ---------------
# Plots
nrows = 2
ncolumns = 2

# Colours
maximum_values_colour = "red"
mean_values_colour = "blue"

# Sizes
fontsize = "large"
labelsize = "large"

####################################################################
# --------------- Datasets reading ---------------
dsb_ds = pandas.read_csv(
    "/PATH/TO/fire_events_dsb.csv",
    delimiter=";",
    decimal=".",
)

firecci51_ds = pandas.read_csv(
    "/PATH/TO/fire_events_final_scandinavia.csv",
    delimiter=",",
    decimal=".",
)

modis_active_fires_ds = pandas.read_csv(
    "/PATH/TO/active_fire_events_scandinavia.csv",
    delimiter=",",
    decimal=".",
)

gfas_ds = pandas.read_csv(
    "/PATH/TO/gfas_events_scandinavia.csv",
    delimiter=",",
    decimal=".",
)

####################################################################
# --------------- Loading variables ---------------
dsb_burned_area = dsb_ds["Burned area"]/10 * 10000
dsb_burned_area = dsb_burned_area.tolist()
dsb_mean_fwi = dsb_ds["Mean fire weather index"].tolist()
dsb_max_fwi = dsb_ds["Max fire weather index"].tolist()
dsb_mean_isi = dsb_ds["Mean initial spread index"].tolist()
dsb_max_isi = dsb_ds["Max initial spread index"].tolist()
dsb_mean_bi = dsb_ds["Mean burning index"].tolist()
dsb_max_bi = dsb_ds["Max burning index"].tolist()
dsb_mean_sc = dsb_ds["Mean spread component"].tolist()
dsb_max_sc = dsb_ds["Max spread component"].tolist()

firecci51_burned_area = firecci51_ds["Burned_area"] * 10000
firecci51_burned_area = firecci51_burned_area.tolist()
firecci51_mean_frp = firecci51_ds["fire_radiative_power_mean"]
firecci51_mean_frp = firecci51_mean_frp.tolist()
firecci51_max_frp = firecci51_ds["fire_radiative_power_max"]
firecci51_max_frp = firecci51_max_frp.tolist()
firecci51_mean_fwi = firecci51_ds["fire_weather_index_mean"].tolist()
firecci51_max_fwi = firecci51_ds["fire_weather_index_max"].tolist()
firecci51_mean_isi = firecci51_ds["initial_spread_index_mean"].tolist()
firecci51_max_isi = firecci51_ds["initial_spread_index_max"].tolist()
firecci51_mean_bui = firecci51_ds["buildup_index_mean"].tolist()
firecci51_max_bui = firecci51_ds["buildup_index_max"].tolist()
firecci51_mean_bi = firecci51_ds["burning_index_mean"].tolist()
firecci51_max_bi = firecci51_ds["burning_index_max"].tolist()
firecci51_mean_sc = firecci51_ds["spread_component_mean"].tolist()
firecci51_max_sc = firecci51_ds["spread_component_max"].tolist()
firecci51_mean_erc = firecci51_ds["energy_release_component_mean"].tolist()
firecci51_max_erc = firecci51_ds["energy_release_component_max"].tolist()

modis_active_fires_frp = modis_active_fires_ds["frp"]
modis_active_fires_frp = modis_active_fires_frp.tolist()
modis_active_fires_fwi = modis_active_fires_ds["fire_weather_index"].tolist()
modis_active_fires_isi = modis_active_fires_ds["initial_spread_index"].tolist()
modis_active_fires_bui = modis_active_fires_ds["buildup_index"].tolist()
modis_active_fires_bi = modis_active_fires_ds["burning_index"].tolist()
modis_active_fires_sc = modis_active_fires_ds["spread_component"].tolist()
modis_active_fires_erc = modis_active_fires_ds["energy_release_component"].tolist()

#gfas_frp = gfas_ds["frpfire"].tolist()
gfas_frp = gfas_ds["frp_megawatts"].tolist()
gfas_fwi = gfas_ds["fire_weather_index"].tolist()
gfas_isi = gfas_ds["initial_spread_index"].tolist()
gfas_bui = gfas_ds["buildup_index"].tolist()
gfas_bi = gfas_ds["burning_index"].tolist()
gfas_sc = gfas_ds["spread_component"].tolist()
gfas_erc = gfas_ds["energy_release_component"].tolist()


####################################################################
logger.info("Plotting end indices")
# --------------- Creating plots ---------------
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')


axs[0,0] = plottings(x_data1 = dsb_burned_area, x_data2 = firecci51_burned_area,
                     y_data1 = dsb_max_fwi, y_data2 = firecci51_max_fwi,
                     dataset1_name = "DSB", dataset2_name = "FireCCI51",
                     index_x = 0, index_y = 0,
                     min_y = -1, max_y = 35, stepsize = 5, plot_number = "a.",
                     legend_label1 = "DSB (n=1991)", legend_label2 = "FireCCI51 (n=467)",
                     x_label = "Burned area (m²)",
                     y_label = "Maximum fire weather index")

axs[0,1] = plottings(x_data1 = dsb_burned_area, x_data2 = firecci51_burned_area,
                     y_data1 = dsb_max_bi, y_data2 = firecci51_max_bi,
                     dataset1_name = "DSB", dataset2_name = "FireCCI51",
                     index_x = 0, index_y = 1,
                     min_y = -1, max_y = 40, stepsize = 5, plot_number = "b.",
                     legend_label1 = "DSB (n=1991)", legend_label2 = "FireCCI51 (n=467)",
                     x_label = "Burned area (m²)",
                     y_label = "Maximum burning index")

axs[1,0] = plottings(x_data1 = gfas_frp, x_data2 = modis_active_fires_frp,
                     y_data1 = gfas_fwi, y_data2 = modis_active_fires_fwi,
                     dataset1_name = "GFAS", dataset2_name = "MODIS Active Fires",
                     index_x = 1, index_y = 0,
                     min_y = -1, max_y = 35, stepsize = 5, plot_number = "c.",
                     legend_label1 = "GFAS (n=6712)", legend_label2 = "MODIS Active Fires (n=9747)",
                     x_label = "Fire radiative power (MW)",
                     y_label = "Fire weather index")

axs[1,1] = plottings(x_data1 = gfas_frp, x_data2 = modis_active_fires_frp,
                     y_data1 = gfas_bi, y_data2 = modis_active_fires_bi,
                     dataset1_name = "GFAS", dataset2_name = "MODIS Active Fires",
                     index_x = 1, index_y = 1,
                     min_y = -1, max_y = 40, stepsize = 5, plot_number = "d.",
                     legend_label1 = "GFAS (n=6712)", legend_label2 = "MODIS Active Fires (n=9747)",
                     x_label = "Fire radiative power (MW)",
                     y_label = "Burning index")

plt.savefig('/PATH/TO/end_indices.png', dpi=500)

########################################################################################################

logger.info("Plotting model components")
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

axs[0,0] = plottings(x_data1 = dsb_burned_area, x_data2 = firecci51_burned_area,
                     y_data1 = dsb_max_isi, y_data2 = firecci51_max_isi,
                     dataset1_name = "DSB", dataset2_name = "FireCCI51",
                     index_x = 0, index_y = 0,
                     min_y = -1, max_y = 25, stepsize = 5, plot_number = "a.",
                     legend_label1 = "DSB (n=1991)", legend_label2 = "FireCCI51 (n=467)",
                    x_label = "Burned area (m²)",
                     y_label = "Maximum initial spread index")

axs[0,1] = plottings(x_data1 = dsb_burned_area, x_data2 = firecci51_burned_area,
                     y_data1 = dsb_max_sc, y_data2 = firecci51_max_sc,
                     dataset1_name = "DSB", dataset2_name = "FireCCI51",
                     index_x = 0, index_y = 1,
                     min_y = -1, max_y = 20, stepsize = 5, plot_number = "b.",
                     legend_label1 = "DSB (n=1991)", legend_label2 = "FireCCI51 (n=467)",
                     x_label = "Burned area (m²)",
                     y_label = "Maximum spread component")

axs[1,0] = plottings(x_data1 = gfas_frp, x_data2 = modis_active_fires_frp,
                     y_data1 = gfas_bui, y_data2 = modis_active_fires_bui,
                     dataset1_name = "GFAS", dataset2_name = "MODIS Active Fires",
                     index_x = 1, index_y = 0,
                     min_y = -5, max_y = 180, stepsize = 15, plot_number = "c.",
                     legend_label1 = "GFAS (n=6712)", legend_label2 = "MODIS Active Fires (n=9747)",
                     x_label = "Fire radiative power (MW)",
                     y_label = "Build-up index")

axs[1,1] = plottings(x_data1 = gfas_frp, x_data2 = modis_active_fires_frp,
                     y_data1 = gfas_erc, y_data2 = modis_active_fires_erc,
                     dataset1_name = "GFAS", dataset2_name = "MODIS Active Fires",
                     index_x = 1, index_y = 1,
                     min_y = -1, max_y = 40, stepsize = 5, plot_number = "d.",
                     legend_label1 = "GFAS (n=6712)", legend_label2 = "MODIS Active Fires (n=9747)",
                     x_label = "Fire radiative power (MW)",
                     y_label = "Energy release component")

plt.savefig('/PATH/TO/individual_components.png', dpi=500)
#plt.show()

########################################################################################################

logger.info("Plotting big fires")
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

dsb_big_fires = dsb_ds[dsb_ds["Burned area"]/10 >= 100]
dsb_big_fires_burned_area = dsb_big_fires["Burned area"]/10 * 10000
dsb_big_fires_burned_area = dsb_big_fires_burned_area.tolist()
dsb_big_fires_max_fwi = dsb_big_fires["Max fire weather index"].tolist()
dsb_big_fires_max_bi = dsb_big_fires["Max burning index"].tolist()

firecci51_big_fires = firecci51_ds[firecci51_ds["Burned_area"] >= 100]
firecci51_big_fires_burned_area = firecci51_big_fires["Burned_area"] * 10000
firecci51_big_fires_burned_area = firecci51_big_fires_burned_area.tolist()
firecci51_big_fires_max_fwi = firecci51_big_fires["fire_weather_index_max"].tolist()
firecci51_big_fires_max_bi = firecci51_big_fires["burning_index_max"].tolist()

axs[0,0] = plottings(x_data1 = dsb_big_fires_burned_area, x_data2 = firecci51_big_fires_burned_area,
                     y_data1 = dsb_big_fires_max_fwi, y_data2 = firecci51_big_fires_max_fwi,
                     dataset1_name = "DSB", dataset2_name = "FireCCI51",
                     index_x = 0, index_y = 0,
                     min_y = -1, max_y = 25, stepsize = 5, plot_number = "a.",
                     legend_label1 = "DSB (n=1991)", legend_label2 = "FireCCI51 (n=467)",
                     x_label = "Burned area (m²)",
                     y_label = "Maximum fire weather index")

axs[0,1] = plottings(x_data1 = dsb_big_fires_burned_area, x_data2 = firecci51_big_fires_burned_area,
                     y_data1 = dsb_big_fires_max_bi, y_data2 = firecci51_big_fires_max_bi,
                     dataset1_name = "DSB", dataset2_name = "FireCCI51",
                     index_x = 0, index_y = 1,
                     min_y = -1, max_y = 20, stepsize = 5, plot_number = "b.",
                     legend_label1 = "DSB (n=1991)", legend_label2 = "FireCCI51 (n=467)",
                     x_label = "Burned area (m²)",
                     y_label = "Maximum burning index")

modis_active_fires_big_fires = modis_active_fires_ds[modis_active_fires_ds["frp"] >= 5]
modis_active_fires_big_fires_frp = modis_active_fires_big_fires["frp"].tolist()
modis_active_fires_big_fires_fwi = modis_active_fires_big_fires["fire_weather_index"].tolist()
modis_active_fires_big_fires_bi = modis_active_fires_big_fires["burning_index"].tolist()

gfas_big_fires = gfas_ds[gfas_ds["frp_megawatts"] >= 5]
gfas_big_fires_frp = gfas_big_fires["frp_megawatts"].tolist()
gfas_big_fires_fwi = gfas_big_fires["fire_weather_index"].tolist()
gfas_big_fires_bi = gfas_big_fires["burning_index"].tolist()

axs[1,0] = plottings(x_data1 = gfas_big_fires_frp, x_data2 = modis_active_fires_big_fires_frp,
                     y_data1 = gfas_big_fires_fwi, y_data2 = modis_active_fires_big_fires_fwi,
                     dataset1_name = "GFAS", dataset2_name = "MODIS Active Fires",
                     index_x = 1, index_y = 0,
                     min_y = -1, max_y = 35, stepsize = 5, plot_number = "c.",
                     legend_label1 = "GFAS (n=6712)", legend_label2 = "MODIS Active Fires (n=9747)",
                     x_label = "Fire radiative power (MW)",
                     y_label = "Fire weather index")

axs[1,1] = plottings(x_data1 = gfas_big_fires_frp, x_data2 = modis_active_fires_big_fires_frp,
                     y_data1 = gfas_big_fires_bi, y_data2 = modis_active_fires_big_fires_bi,
                     dataset1_name = "GFAS", dataset2_name = "MODIS Active Fires",
                     index_x = 1, index_y = 1,
                     min_y = -1, max_y = 40, stepsize = 5, plot_number = "d.",
                     legend_label1 = "GFAS (n=6712)", legend_label2 = "MODIS Active Fires (n=9747)",
                     x_label = "Fire radiative power (MW)",
                     y_label = "Burning index")

plt.savefig('/PATH/TO/big_fires.png', dpi=500)
#plt.show()

########################################################################################################

logger.info("Plotting burned area against radiative power")
fig, axs = plt.subplots(nrows = 1, ncols = 1,
                        figsize = (10,8), layout='tight')

x_data1 = firecci51_burned_area
y_data1 = firecci51_mean_frp

sort_dataset1 = np.argsort(x_data1)
dataset1_x_sorted = np.array(x_data1)[sort_dataset1]
    
dataset1_range = np.logspace(np.log10(dataset1_x_sorted.min()),
                             np.log10(dataset1_x_sorted.max()),
                             200)

dataset1_y_sorted = np.array(y_data1)[sort_dataset1]

dataset1_smooth, r_squared_dataset1, p_value1, residuals1, x_log1 = get_log_linear_stats(dataset1_range,
                                                           dataset1_x_sorted,
                                                           dataset1_y_sorted)

scatter_factor1 = 10 ** np.std(residuals1)

plot1 = axs.scatter(
    x=x_data1,
    y=y_data1,
    c="salmon",
    s=10,
    marker="v",
    alpha=0.5,
)

axs.plot(dataset1_range, dataset1_smooth, "r-",
         label=f"Power trendline\nR²={format_decimals(r_squared_dataset1)}, p={format_decimals(p_value1)}, scatter≈x{scatter_factor1:.2f}")
    
axs.set_xscale("log")
axs.set_yscale("log")
#axs.set_ylim([-0.1,1.5])
#axs.yaxis.set_ticks(np.arange(0,6+1,1))
axs.legend(loc=0)
axs.set_xlabel("Burned area (m²)", size=fontsize)
axs.set_ylabel("Radiative Power (MW)", size=fontsize)

plt.savefig('/PATH/TO/burnedarea_vs_radiativepower.png', dpi=500)
#plt.show()


########################################################################################################

logger.info("Starting the DSB burned area plotting...")

# --------------- Creating plots ---------------
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

sort_dsb_burned_area = np.argsort(dsb_burned_area)
dsb_burned_area_sorted = np.array(dsb_burned_area)[sort_dsb_burned_area]
dsb_burned_area_range = np.logspace(np.log10(dsb_burned_area_sorted.min()),
                                    np.log10(dsb_burned_area_sorted.max()),
                                    200)

# ----------------------------------------------------------------------------------------------------

dsb_max_fwi_sorted = np.array(dsb_max_fwi)[sort_dsb_burned_area]
dsb_mean_fwi_sorted = np.array(dsb_mean_fwi)[sort_dsb_burned_area]

dsb_max_fwi_smooth, r_squared_fwi_max = get_log_linear_stats(dsb_burned_area_range,
                                                             dsb_burned_area_sorted,
                                                             dsb_max_fwi_sorted)
dsb_mean_fwi_smooth, r_squared_fwi_mean = get_log_linear_stats(dsb_burned_area_range,
                                                               dsb_burned_area_sorted,
                                                               dsb_mean_fwi_sorted)

dsb_fwi_max = axs[0, 0].scatter(
    x=dsb_burned_area,
    y=dsb_max_fwi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum fire weather index",
)
axs[0,0].plot(dsb_burned_area_range, dsb_max_fwi_smooth, "r--", label =
              f"Power trendline (Max FWI)\nR2={round(r_squared_fwi_max,3)}")

dsb_fwi_mean = axs[0, 0].scatter(
    x=dsb_burned_area,
    y=dsb_mean_fwi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean fire weather index",
)
axs[0,0].plot(dsb_burned_area_range, dsb_mean_fwi_smooth, "b-", label =
              f"Power trendline (Mean FWI)\nR2={round(r_squared_fwi_mean,3)}")
axs[0,0].set_xscale("log")
axs[0,0].legend(loc=0)
axs[0,0].set_xlabel("Burned area (ha)", size=fontsize)
axs[0,0].set_ylabel("Fire weather index", size=fontsize)
start, end = 0, 35
axs[0,0].set_ylim([start,end])
axs[0,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,0].set_title("a.", loc="left")

# ----------------------------------------------------------------------------------------------------
dsb_max_bi_sorted = np.array(dsb_max_bi)[sort_dsb_burned_area]
dsb_mean_bi_sorted = np.array(dsb_mean_bi)[sort_dsb_burned_area]

dsb_max_bi_smooth, r_squared_bi_max = get_log_linear_stats(dsb_burned_area_range,
                                                           dsb_burned_area_sorted,
                                                           dsb_max_bi_sorted)
dsb_mean_bi_smooth, r_squared_bi_mean = get_log_linear_stats(dsb_burned_area_range,
                                                             dsb_burned_area_sorted,
                                                             dsb_mean_bi_sorted)

dsb_bi_max = axs[0, 1].scatter(
    x=dsb_burned_area,
    y=dsb_max_bi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum burning index",
)
axs[0,1].plot(dsb_burned_area_range, dsb_max_bi_smooth, "r--", label =
              f"Power trendline (Max BI)\nR2={round(r_squared_bi_max,3)}")

dsb_bi_mean = axs[0, 1].scatter(
    x=dsb_burned_area,
    y=dsb_mean_bi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean burning index",
)
axs[0,1].plot(dsb_burned_area_range, dsb_mean_bi_smooth, "b-", label =
              f"Power trendline (Mean BI)\nR2={round(r_squared_bi_mean,3)}")
axs[0,1].set_xscale("log")
axs[0,1].set_xscale("log")
axs[0,1].legend(loc=0)
axs[0,1].set_xlabel("Burned area (ha)", size=fontsize)
axs[0,1].set_ylabel("NFDRS Burning index", size=fontsize)
start, end = 0, 45
axs[0,1].set_ylim([start,end])
axs[0,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,1].set_title("b.", loc="left")

# ----------------------------------------------------------------------------------------------------
dsb_max_isi_sorted = np.array(dsb_max_isi)[sort_dsb_burned_area]
dsb_mean_isi_sorted = np.array(dsb_mean_isi)[sort_dsb_burned_area]

dsb_max_isi_smooth, r_squared_isi_max = get_log_linear_stats(dsb_burned_area_range,
                                                             dsb_burned_area_sorted,
                                                             dsb_max_isi_sorted)
dsb_mean_isi_smooth, r_squared_isi_mean = get_log_linear_stats(dsb_burned_area_range,
                                                               dsb_burned_area_sorted,
                                                               dsb_mean_isi_sorted)

dsb_isi_max = axs[1, 0].scatter(
    x=dsb_burned_area,
    y=dsb_max_isi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum initial spread index",
)
axs[1,0].plot(dsb_burned_area_range, dsb_max_isi_smooth, "r--", label =
              f"Power trendline (Max ISI)\nR2={round(r_squared_isi_max,3)}")

dsb_isi_mean = axs[1, 0].scatter(
    x=dsb_burned_area,
    y=dsb_mean_isi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean initial spread index",
)
axs[1,0].plot(dsb_burned_area_range, dsb_mean_isi_smooth, "b-", label =
              f"Power trendline (Mean ISI)\nR2={round(r_squared_isi_mean,3)}")
axs[1,0].set_xscale("log")
axs[1,0].legend(loc=0)
axs[1,0].set_xlabel("Burned area (ha)", size=fontsize)
axs[1,0].set_ylabel("FWI Initial spread index", size=fontsize)
start, end = 0, 25
axs[1,0].set_ylim([start,end])
axs[1,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[1,0].set_title("c.", loc="left")

# ----------------------------------------------------------------------------------------------------
dsb_max_sc_sorted = np.array(dsb_max_sc)[sort_dsb_burned_area]
dsb_mean_sc_sorted = np.array(dsb_mean_sc)[sort_dsb_burned_area]

dsb_max_sc_smooth, r_squared_sc_max = get_log_linear_stats(dsb_burned_area_range,
                                                           dsb_burned_area_sorted,
                                                           dsb_max_sc_sorted)
dsb_mean_sc_smooth, r_squared_sc_mean = get_log_linear_stats(dsb_burned_area_range,
                                                             dsb_burned_area_sorted,
                                                             dsb_mean_sc_sorted)

dsb_sc_max = axs[1, 1].scatter(
    x=dsb_burned_area,
    y=dsb_max_sc,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum spread component",
)
axs[1,1].plot(dsb_burned_area_range, dsb_max_sc_smooth, "r--", label =
              f"Power trendline (Max SC)\nR2={round(r_squared_sc_max,4)}")

dsb_sc_mean = axs[1, 1].scatter(
    x=dsb_burned_area,
    y=dsb_mean_sc,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean spread component",
)
axs[1,1].plot(dsb_burned_area_range, dsb_mean_sc_smooth, "b-", label =
              f"Power trendline (Mean SC)\nR2={round(r_squared_sc_mean,3)}")
axs[1,1].set_xscale("log")
axs[1,1].set_xscale("log")
axs[1,1].legend(loc=0)
axs[1,1].set_xlabel("Burned area (ha)", size=fontsize)
axs[1,1].set_ylabel("NFDRS Spread component", size=fontsize)
start, end = 0, 20
axs[1,1].set_ylim([start,end])
axs[1,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[1,1].set_title("d.", loc="left")

plt.savefig('/PATH/TO/DSB_burned_area.png', dpi=500)
#plt.show()

####################################################################
logger.info("Starting the FireCCI51 burned area plotting...")

# --------------- Creating plots ---------------
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

sort_firecci51_burned_area = np.argsort(firecci51_burned_area)
firecci51_burned_area_sorted = np.array(firecci51_burned_area)[sort_firecci51_burned_area]
firecci51_burned_area_range = np.logspace(np.log10(firecci51_burned_area_sorted.min()),
                                    np.log10(firecci51_burned_area_sorted.max()),
                                    200)

# ----------------------------------------------------------------------------------------------------

firecci51_max_fwi_sorted = np.array(firecci51_max_fwi)[sort_firecci51_burned_area]
firecci51_mean_fwi_sorted = np.array(firecci51_mean_fwi)[sort_firecci51_burned_area]

firecci51_max_fwi_smooth, r_squared_fwi_max = get_log_linear_stats(firecci51_burned_area_range,
                                                                   firecci51_burned_area_sorted,
                                                                   firecci51_max_fwi_sorted)
firecci51_mean_fwi_smooth, r_squared_fwi_mean = get_log_linear_stats(firecci51_burned_area_range,
                                                                     firecci51_burned_area_sorted,
                                                                     firecci51_mean_fwi_sorted)

firecci51_fwi_max = axs[0, 0].scatter(
    x=firecci51_burned_area,
    y=firecci51_max_fwi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum fire weather index",
)
axs[0,0].plot(firecci51_burned_area_range, firecci51_max_fwi_smooth, "r--", label =
              f"Power trendline (Max FWI)\nR2={round(r_squared_fwi_max,3)}")

firecci51_fwi_mean = axs[0, 0].scatter(
    x=firecci51_burned_area,
    y=firecci51_mean_fwi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean fire weather index",
)
axs[0,0].plot(firecci51_burned_area_range, firecci51_mean_fwi_smooth, "b-", label =
              f"Power trendline (Mean FWI)\nR2={round(r_squared_fwi_mean,3)}")
axs[0,0].set_xscale("log")
axs[0,0].legend(loc=0)
axs[0,0].set_xlabel("Burned area (ha)", size=fontsize)
axs[0,0].set_ylabel("FWI Fire weather index", size=fontsize)
start, end = 0, 20
axs[0,0].set_ylim([start,end])
axs[0,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,0].set_title("a.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_bi_sorted = np.array(firecci51_max_bi)[sort_firecci51_burned_area]
firecci51_mean_bi_sorted = np.array(firecci51_mean_bi)[sort_firecci51_burned_area]

firecci51_max_bi_smooth, r_squared_bi_max = get_log_linear_stats(firecci51_burned_area_range,
                                                                 firecci51_burned_area_sorted,
                                                                 firecci51_max_bi_sorted)
firecci51_mean_bi_smooth, r_squared_bi_mean = get_log_linear_stats(firecci51_burned_area_range,
                                                                   firecci51_burned_area_sorted,
                                                                   firecci51_mean_bi_sorted)

firecci51_bi_max = axs[0, 1].scatter(
    x=firecci51_burned_area,
    y=firecci51_max_bi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum burning index",
)
axs[0,1].plot(firecci51_burned_area_range, firecci51_max_bi_smooth, "r--", label =
              f"Power trendline (Max BI)\nR2={round(r_squared_bi_max,3)}")

firecci51_bi_mean = axs[0, 1].scatter(
    x=firecci51_burned_area,
    y=firecci51_mean_bi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean burning index",
)
axs[0,1].plot(firecci51_burned_area_range, firecci51_mean_bi_smooth, "b-", label =
              f"Power trendline (Mean BI)\nR2={round(r_squared_bi_mean,3)}")
axs[0,1].set_xscale("log")
axs[0,1].set_xscale("log")
axs[0,1].legend(loc=0)
axs[0,1].set_xlabel("Burned area (ha)", size=fontsize)
axs[0,1].set_ylabel("NFDRS Burning index", size=fontsize)
start, end = 0, 45
axs[0,1].set_ylim([start,end])
axs[0,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,1].set_title("b.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_isi_sorted = np.array(firecci51_max_isi)[sort_firecci51_burned_area]
firecci51_mean_isi_sorted = np.array(firecci51_mean_isi)[sort_firecci51_burned_area]

firecci51_max_isi_smooth, r_squared_isi_max = get_log_linear_stats(firecci51_burned_area_range,
                                                                   firecci51_burned_area_sorted,
                                                                   firecci51_max_isi_sorted)
firecci51_mean_isi_smooth, r_squared_isi_mean = get_log_linear_stats(firecci51_burned_area_range,
                                                                     firecci51_burned_area_sorted,
                                                                     firecci51_mean_isi_sorted)

firecci51_isi_max = axs[1, 0].scatter(
    x=firecci51_burned_area,
    y=firecci51_max_isi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum initial spread index",
)
axs[1,0].plot(firecci51_burned_area_range, firecci51_max_isi_smooth, "r--", label =
              f"Power trendline (Max ISI)\nR2={round(r_squared_isi_max,3)}")

firecci51_isi_mean = axs[1, 0].scatter(
    x=firecci51_burned_area,
    y=firecci51_mean_isi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean initial spread index",
)
axs[1,0].plot(firecci51_burned_area_range, firecci51_mean_isi_smooth, "b-", label =
              f"Power trendline (Mean ISI)\nR2={round(r_squared_isi_mean,3)}")
axs[1,0].set_xscale("log")
axs[1,0].legend(loc=0)
axs[1,0].set_xlabel("Burned area (ha)", size=fontsize)
axs[1,0].set_ylabel("FWI Initial spread index", size=fontsize)
start, end = 0, 8
axs[1,0].set_ylim([start,end])
axs[1,0].yaxis.set_ticks(np.arange(start,end+1,2))
axs[1,0].set_title("c.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_sc_sorted = np.array(firecci51_max_sc)[sort_firecci51_burned_area]
firecci51_mean_sc_sorted = np.array(firecci51_mean_sc)[sort_firecci51_burned_area]

firecci51_max_sc_smooth, r_squared_sc_max = get_log_linear_stats(firecci51_burned_area_range,
                                                                 firecci51_burned_area_sorted,
                                                                 firecci51_max_sc_sorted)
firecci51_mean_sc_smooth, r_squared_sc_mean = get_log_linear_stats(firecci51_burned_area_range,
                                                                   firecci51_burned_area_sorted,
                                                                   firecci51_mean_sc_sorted)

firecci51_sc_max = axs[1, 1].scatter(
    x=firecci51_burned_area,
    y=firecci51_max_sc,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum spread component",
)
axs[1,1].plot(firecci51_burned_area_range, firecci51_max_sc_smooth, "r--", label =
              f"Power trendline (Max SC)\nR2={round(r_squared_sc_max,3)}")

firecci51_sc_mean = axs[1, 1].scatter(
    x=firecci51_burned_area,
    y=firecci51_mean_sc,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean spread component",
)
axs[1,1].plot(firecci51_burned_area_range, firecci51_mean_sc_smooth, "b-", label =
              f"Power trendline (Mean SC)\nR2={round(r_squared_sc_mean,3)}")
axs[1,1].set_xscale("log")
axs[1,1].set_xscale("log")
axs[1,1].legend(loc=0)
axs[1,1].set_xlabel("Burned area (ha)", size=fontsize)
axs[1,1].set_ylabel("NFDRS Spread component", size=fontsize)
start, end = 0, 12
axs[1,1].set_ylim([start,end])
axs[1,1].yaxis.set_ticks(np.arange(start,end+1,2))
axs[1,1].set_title("d.", loc="left")

plt.savefig('/PATH/TO/firecci51_burned_area.png', dpi=500)
#plt.show()

####################################################################
logger.info("Starting the FireCCI51 max fire radiative power plotting...")

# --------------- Creating plots ---------------
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

sort_firecci51_max_frp = np.argsort(firecci51_max_frp)
firecci51_max_frp_sorted = np.array(firecci51_max_frp)[sort_firecci51_max_frp]
firecci51_max_frp_range = np.logspace(np.log10(firecci51_max_frp_sorted.min()),
                                    np.log10(firecci51_max_frp_sorted.max()),
                                    200)

# ----------------------------------------------------------------------------------------------------

firecci51_max_fwi_sorted = np.array(firecci51_max_fwi)[sort_firecci51_max_frp]
firecci51_mean_fwi_sorted = np.array(firecci51_mean_fwi)[sort_firecci51_max_frp]

log_firecci51_max_fwi = np.log10(firecci51_max_fwi_sorted)
log_firecci51_mean_fwi = np.log10(firecci51_mean_fwi_sorted)

firecci51_max_fwi_smooth, r_squared_fwi_max = get_log_linear_stats(firecci51_max_frp_range,
                                                                   firecci51_max_frp_sorted,
                                                                   firecci51_max_fwi_sorted)
firecci51_mean_fwi_smooth, r_squared_fwi_mean = get_log_linear_stats(firecci51_max_frp_range,
                                                                     firecci51_max_frp_sorted,
                                                                     firecci51_mean_fwi_sorted)

firecci51_fwi_max = axs[0, 0].scatter(
    x=firecci51_max_frp,
    y=firecci51_max_fwi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum fire weather index",
)
axs[0,0].plot(firecci51_max_frp_range, firecci51_max_fwi_smooth, "r--", label =
              f"Power trendline (Max FWI)\nR2={round(r_squared_fwi_max,3)}")

firecci51_fwi_mean = axs[0, 0].scatter(
    x=firecci51_max_frp,
    y=firecci51_mean_fwi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean fire weather index",
)
axs[0,0].plot(firecci51_max_frp_range, firecci51_mean_fwi_smooth, "b-", label =
              f"Power trendline (Mean FWI)\nR2={round(r_squared_fwi_mean,3)}")
axs[0,0].set_xscale("log")
axs[0,0].legend(loc=0)
axs[0,0].set_xlabel("Max fire radiative power (W/m2)", size=fontsize)
axs[0,0].set_ylabel("FWI Fire weather index", size=fontsize)
start, end = 0, 20
axs[0,0].set_ylim([start,end])
axs[0,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,0].set_title("a.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_bi_sorted = np.array(firecci51_max_bi)[sort_firecci51_max_frp]
firecci51_mean_bi_sorted = np.array(firecci51_mean_bi)[sort_firecci51_max_frp]

firecci51_max_bi_smooth, r_squared_bi_max = get_log_linear_stats(firecci51_max_frp_range,
                                                                 firecci51_max_frp_sorted,
                                                                 firecci51_max_bi_sorted)
firecci51_mean_bi_smooth, r_squared_bi_mean = get_log_linear_stats(firecci51_max_frp_range,
                                                                   firecci51_max_frp_sorted,
                                                                   firecci51_mean_bi_sorted)

firecci51_bi_max = axs[0, 1].scatter(
    x=firecci51_max_frp,
    y=firecci51_max_bi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum burning index",
)
axs[0,1].plot(firecci51_max_frp_range, firecci51_max_bi_smooth, "r--", label =
              f"Power trendline (Max BI)\nR2={round(r_squared_bi_max,4)}")

firecci51_bi_mean = axs[0, 1].scatter(
    x=firecci51_max_frp,
    y=firecci51_mean_bi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean burning index",
)
axs[0,1].plot(firecci51_max_frp_range, firecci51_mean_bi_smooth, "b-", label =
              f"Power trendline (Mean BI)\nR2={round(r_squared_bi_mean,3)}")
axs[0,1].set_xscale("log")
axs[0,1].set_xscale("log")
axs[0,1].legend(loc=0)
axs[0,1].set_xlabel("Max fire radiative power (W/m2)", size=fontsize)
axs[0,1].set_ylabel("NFDRS Burning index", size=fontsize)
start, end = 0, 45
axs[0,1].set_ylim([start,end])
axs[0,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,1].set_title("b.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_bui_sorted = np.array(firecci51_max_bui)[sort_firecci51_max_frp]
firecci51_mean_bui_sorted = np.array(firecci51_mean_bui)[sort_firecci51_max_frp]

firecci51_max_bui_smooth, r_squared_bui_max = get_log_linear_stats(firecci51_max_frp_range,
                                                                   firecci51_max_frp_sorted,
                                                                   firecci51_max_bui_sorted)
firecci51_mean_bui_smooth, r_squared_bui_mean = get_log_linear_stats(firecci51_max_frp_range,
                                                                     firecci51_max_frp_sorted,
                                                                     firecci51_mean_bui_sorted)

firecci51_bui_max = axs[1, 0].scatter(
    x=firecci51_max_frp,
    y=firecci51_max_bui,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum buildup index",
)
axs[1,0].plot(firecci51_max_frp_range, firecci51_max_bui_smooth, "r--", label =
              f"Power trendline (Max BUI)\nR2={round(r_squared_bui_max,3)}")

firecci51_bui_mean = axs[1, 0].scatter(
    x=firecci51_max_frp,
    y=firecci51_mean_bui,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean buildup index",
)
axs[1,0].plot(firecci51_max_frp_range, firecci51_mean_bui_smooth, "b-", label =
              f"Power trendline (Mean BUI)\nR2={round(r_squared_bui_mean,3)}")
axs[1,0].set_xscale("log")
axs[1,0].legend(loc=0)
axs[1,0].set_xlabel("Max fire radiative power (W/m2)", size=fontsize)
axs[1,0].set_ylabel("FWI Buildup index", size=fontsize)
start, end = 0, 75
axs[1,0].set_ylim([start,end])
axs[1,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[1,0].set_title("c.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_erc_sorted = np.array(firecci51_max_sc)[sort_firecci51_max_frp]
firecci51_mean_erc_sorted = np.array(firecci51_mean_sc)[sort_firecci51_max_frp]

firecci51_max_erc_smooth, r_squared_erc_max = get_log_linear_stats(firecci51_max_frp_range,
                                                                   firecci51_max_frp_sorted,
                                                                   firecci51_max_erc_sorted)
firecci51_mean_erc_smooth, r_squared_erc_mean = get_log_linear_stats(firecci51_max_frp_range,
                                                                     firecci51_max_frp_sorted,
                                                                     firecci51_mean_erc_sorted)

firecci51_erc_max = axs[1, 1].scatter(
    x=firecci51_max_frp,
    y=firecci51_max_sc,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum energy release component",
)
axs[1,1].plot(firecci51_max_frp_range, firecci51_max_erc_smooth, "r--", label =
              f"Power trendline (Max ERC)\nR2={round(r_squared_erc_max,4)}")

firecci51_erc_mean = axs[1, 1].scatter(
    x=firecci51_max_frp,
    y=firecci51_mean_sc,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean energy release component",
)
axs[1,1].plot(firecci51_max_frp_range, firecci51_mean_erc_smooth, "b-", label =
              f"Power trendline (Mean ERC)\nR2={round(r_squared_erc_mean,3)}")
axs[1,1].set_xscale("log")
axs[1,1].set_xscale("log")
axs[1,1].legend(loc=0)
axs[1,1].set_xlabel("Max fire radiative power (W/m2)", size=fontsize)
axs[1,1].set_ylabel("NFDRS Energy release component", size=fontsize)
start, end = 0, 12
axs[1,1].set_ylim([start,end])
axs[1,1].yaxis.set_ticks(np.arange(start,end+1,2))
axs[1,1].set_title("d.", loc="left")

plt.savefig('/PATH/TO/firecci51_max_frp.png', dpi=500)
#plt.show()

####################################################################
logger.info("Starting the FireCCI51 mean fire radiative power plotting...")

# --------------- Creating plots ---------------
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

sort_firecci51_mean_frp = np.argsort(firecci51_mean_frp)
firecci51_mean_frp_sorted = np.array(firecci51_mean_frp)[sort_firecci51_mean_frp]
firecci51_mean_frp_range = np.logspace(np.log10(firecci51_mean_frp_sorted.min()),
                                    np.log10(firecci51_mean_frp_sorted.max()),
                                    200)

# ----------------------------------------------------------------------------------------------------

firecci51_max_fwi_sorted = np.array(firecci51_max_fwi)[sort_firecci51_mean_frp]
firecci51_mean_fwi_sorted = np.array(firecci51_mean_fwi)[sort_firecci51_mean_frp]

log_firecci51_max_fwi = np.log10(firecci51_max_fwi_sorted)
log_firecci51_mean_fwi = np.log10(firecci51_mean_fwi_sorted)

firecci51_max_fwi_smooth, r_squared_fwi_max = get_log_linear_stats(firecci51_mean_frp_range,
                                                                   firecci51_mean_frp_sorted,
                                                                   firecci51_max_fwi_sorted)
firecci51_mean_fwi_smooth, r_squared_fwi_mean = get_log_linear_stats(firecci51_mean_frp_range,
                                                                     firecci51_mean_frp_sorted,
                                                                     firecci51_mean_fwi_sorted)

firecci51_fwi_max = axs[0, 0].scatter(
    x=firecci51_mean_frp,
    y=firecci51_max_fwi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum fire weather index",
)
axs[0,0].plot(firecci51_mean_frp_range, firecci51_max_fwi_smooth, "r--", label =
              f"Power trendline (Max FWI)\nR2={round(r_squared_fwi_max,3)}")

firecci51_fwi_mean = axs[0, 0].scatter(
    x=firecci51_mean_frp,
    y=firecci51_mean_fwi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean fire weather index",
)
axs[0,0].plot(firecci51_mean_frp_range, firecci51_mean_fwi_smooth, "b-", label =
              f"Power trendline (Mean FWI)\nR2={round(r_squared_fwi_mean,3)}")
axs[0,0].set_xscale("log")
axs[0,0].legend(loc=0)
axs[0,0].set_xlabel("Mean fire radiative power (W/m2)", size=fontsize)
axs[0,0].set_ylabel("FWI Fire weather index", size=fontsize)
start, end = 0, 20
axs[0,0].set_ylim([start,end])
axs[0,0].yaxis.set_ticks(np.arange(start,end+1,2))
axs[0,0].set_title("a.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_bi_sorted = np.array(firecci51_max_bi)[sort_firecci51_mean_frp]
firecci51_mean_bi_sorted = np.array(firecci51_mean_bi)[sort_firecci51_mean_frp]

firecci51_max_bi_smooth, r_squared_bi_max = get_log_linear_stats(firecci51_mean_frp_range,
                                                                 firecci51_mean_frp_sorted,
                                                                 firecci51_max_bi_sorted)
firecci51_mean_bi_smooth, r_squared_bi_mean = get_log_linear_stats(firecci51_mean_frp_range,
                                                                   firecci51_mean_frp_sorted,
                                                                   firecci51_mean_bi_sorted)

firecci51_bi_max = axs[0, 1].scatter(
    x=firecci51_mean_frp,
    y=firecci51_max_bi,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum burning index",
)
axs[0,1].plot(firecci51_mean_frp_range, firecci51_max_bi_smooth, "r--", label =
              f"Power trendline (Max BI)\nR2={round(r_squared_bi_max,3)}")

firecci51_bi_mean = axs[0, 1].scatter(
    x=firecci51_mean_frp,
    y=firecci51_mean_bi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean burning index",
)
axs[0,1].plot(firecci51_mean_frp_range, firecci51_mean_bi_smooth, "b-", label =
              f"Power trendline (Mean BI)\nR2={round(r_squared_bi_mean,3)}")
axs[0,1].set_xscale("log")
axs[0,1].set_xscale("log")
axs[0,1].legend(loc=0)
axs[0,1].set_xlabel("Mean fire radiative power (W/m2)", size=fontsize)
axs[0,1].set_ylabel("NFDRS Burning index", size=fontsize)
start, end = 0, 45
axs[0,1].set_ylim([start,end])
axs[0,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,1].set_title("b.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_bui_sorted = np.array(firecci51_max_bui)[sort_firecci51_mean_frp]
firecci51_mean_bui_sorted = np.array(firecci51_mean_bui)[sort_firecci51_mean_frp]

firecci51_max_bui_smooth, r_squared_bui_max = get_log_linear_stats(firecci51_mean_frp_range,
                                                                   firecci51_mean_frp_sorted,
                                                                   firecci51_max_bui_sorted)
firecci51_mean_bui_smooth, r_squared_bui_mean = get_log_linear_stats(firecci51_mean_frp_range,
                                                                     firecci51_mean_frp_sorted,
                                                                     firecci51_mean_bui_sorted)

firecci51_bui_max = axs[1, 0].scatter(
    x=firecci51_mean_frp,
    y=firecci51_max_bui,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum buildup index",
)
axs[1,0].plot(firecci51_mean_frp_range, firecci51_max_bui_smooth, "r--", label =
              f"Power trendline (Max BUI)\nR2={round(r_squared_bui_max,3)}")

firecci51_bui_mean = axs[1, 0].scatter(
    x=firecci51_mean_frp,
    y=firecci51_mean_bui,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean buildup index",
)
axs[1,0].plot(firecci51_mean_frp_range, firecci51_mean_bui_smooth, "b-", label =
              f"Power trendline (Mean BUI)\nR2={round(r_squared_bui_mean,3)}")
axs[1,0].set_xscale("log")
axs[1,0].legend(loc=0)
axs[1,0].set_xlabel("Mean fire radiative power (W/m2)", size=fontsize)
axs[1,0].set_ylabel("FWI Buildup index", size=fontsize)
start, end = 0, 75
axs[1,0].set_ylim([start,end])
axs[1,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[1,0].set_title("c.", loc="left")

# ----------------------------------------------------------------------------------------------------
firecci51_max_erc_sorted = np.array(firecci51_max_erc)[sort_firecci51_mean_frp]
firecci51_mean_erc_sorted = np.array(firecci51_mean_erc)[sort_firecci51_mean_frp]

firecci51_max_erc_smooth, r_squared_erc_max = get_log_linear_stats(firecci51_mean_frp_range,
                                                                   firecci51_mean_frp_sorted,
                                                                   firecci51_max_erc_sorted)
firecci51_mean_erc_smooth, r_squared_erc_mean = get_log_linear_stats(firecci51_mean_frp_range,
                                                                     firecci51_mean_frp_sorted,
                                                                     firecci51_mean_erc_sorted)

firecci51_erc_max = axs[1, 1].scatter(
    x=firecci51_mean_frp,
    y=firecci51_max_erc,
    c=maximum_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Maximum energy release component",
)
axs[1,1].plot(firecci51_mean_frp_range, firecci51_max_erc_smooth, "r--", label =
              f"Power trendline (Max ERC)\nR2={round(r_squared_erc_max,3)}")

firecci51_erc_mean = axs[1, 1].scatter(
    x=firecci51_mean_frp,
    y=firecci51_mean_erc,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Mean energy release component",
)
axs[1,1].plot(firecci51_mean_frp_range, firecci51_mean_erc_smooth, "b-", label =
              f"Power trendline (Mean ERC)\nR2={round(r_squared_erc_mean,3)}")
axs[1,1].set_xscale("log")
axs[1,1].set_xscale("log")
axs[1,1].legend(loc=0)
axs[1,1].set_xlabel("Mean fire radiative power (W/m2)", size=fontsize)
axs[1,1].set_ylabel("NFDRS Energy release component", size=fontsize)
start, end = 0, 35
axs[1,1].set_ylim([start,end])
axs[1,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[1,1].set_title("d.", loc="left")

plt.savefig('/PATH/TO/firecci51_mean_frp.png', dpi=500)
#plt.show()

####################################################################
logger.info("Starting the MODIS Active Fires fire radiative power plotting...")

# --------------- Creating plots ---------------
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

sort_modis_active_fires_frp = np.argsort(modis_active_fires_frp)
modis_active_fires_frp_sorted = np.array(modis_active_fires_frp)[sort_modis_active_fires_frp]
modis_active_fires_frp_range = np.logspace(np.log10(modis_active_fires_frp_sorted.min()),
                                    np.log10(modis_active_fires_frp_sorted.max()),
                                    200)

# ----------------------------------------------------------------------------------------------------

modis_active_fires_fwi_sorted = np.array(modis_active_fires_fwi)[sort_modis_active_fires_frp]

log_modis_active_fires_fwi = np.log10(modis_active_fires_fwi_sorted)

modis_active_fires_fwi_smooth, r_squared_fwi = get_log_linear_stats(modis_active_fires_frp_range,
                                                                   modis_active_fires_frp_sorted,
                                                                   modis_active_fires_fwi_sorted)

modis_active_fires_fwi = axs[0, 0].scatter(
    x=modis_active_fires_frp,
    y=modis_active_fires_fwi,
    c=mean_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Fire weather index",
)
axs[0,0].plot(modis_active_fires_frp_range, modis_active_fires_fwi_smooth, "r--", label =
              f"Power trendline (FWI)\nR2={round(r_squared_fwi,5)}")


axs[0,0].set_xscale("log")
axs[0,0].legend(loc=0)
axs[0,0].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[0,0].set_ylabel("FWI Fire weather index", size=fontsize)
start, end = 0, 35
#axs[0,0].set_ylim([start,end])
axs[0,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,0].set_title("a.", loc="left")

# ----------------------------------------------------------------------------------------------------
modis_active_fires_bi_sorted = np.array(modis_active_fires_bi)[sort_modis_active_fires_frp]

modis_active_fires_bi_smooth, r_squared_bi = get_log_linear_stats(modis_active_fires_frp_range,
                                                                 modis_active_fires_frp_sorted,
                                                                 modis_active_fires_bi_sorted)

modis_active_fires_bi = axs[0, 1].scatter(
    x=modis_active_fires_frp,
    y=modis_active_fires_bi,
    c=mean_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Burning index",
)
axs[0,1].plot(modis_active_fires_frp_range, modis_active_fires_bi_smooth, "r--", label =
              f"Power trendline (BI)\nR2={round(r_squared_bi,3)}")


axs[0,1].set_xscale("log")
axs[0,1].legend(loc=0)
axs[0,1].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[0,1].set_ylabel("NFDRS Burning index", size=fontsize)
start, end = 0, 40
axs[0,1].set_ylim([start,end])
axs[0,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,1].set_title("b.", loc="left")

# ----------------------------------------------------------------------------------------------------
modis_active_fires_bui_sorted = np.array(modis_active_fires_bui)[sort_modis_active_fires_frp]

modis_active_fires_bui_smooth, r_squared_bui = get_log_linear_stats(modis_active_fires_frp_range,
                                                                     modis_active_fires_frp_sorted,
                                                                     modis_active_fires_bui_sorted)

modis_active_fires_bui = axs[1, 0].scatter(
    x=modis_active_fires_frp,
    y=modis_active_fires_bui,
    c=mean_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Buildup index",
)
axs[1,0].plot(modis_active_fires_frp_range, modis_active_fires_bui_smooth, "r--", label =
              f"Power trendline (BUI)\nR2={round(r_squared_bui,3)}")

axs[1,0].set_xscale("log")
axs[1,0].legend(loc=0)
axs[1,0].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[1,0].set_ylabel("FWI Buildup index", size=fontsize)
start, end = 0, 120
axs[1,0].set_ylim([start,end])
axs[1,0].yaxis.set_ticks(np.arange(start,end+1,10))
axs[1,0].set_title("c.", loc="left")

# ----------------------------------------------------------------------------------------------------
modis_active_fires_erc_sorted = np.array(modis_active_fires_erc)[sort_modis_active_fires_frp]

modis_active_fires_erc_smooth, r_squared_erc = get_log_linear_stats(modis_active_fires_frp_range,
                                                                     modis_active_fires_frp_sorted,
                                                                     modis_active_fires_erc_sorted)

modis_active_fires_erc = axs[1, 1].scatter(
    x=modis_active_fires_frp,
    y=modis_active_fires_erc,
    c=mean_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Energy release component",
)
axs[1,1].plot(modis_active_fires_frp_range, modis_active_fires_erc_smooth, "r--", label =
              f"Power trendline (ERC)\nR2={round(r_squared_erc,3)}")

axs[1,1].set_xscale("log")
axs[1,1].legend(loc=0)
axs[1,1].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[1,1].set_ylabel("NFDRS Energy release component", size=fontsize)
start, end = 0, 35
axs[1,1].set_ylim([start,end])
axs[1,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[1,1].set_title("d.", loc="left")

plt.savefig('/PATH/TO/modis_active_fires_frp.png', dpi=500)
#plt.show()

####################################################################
logger.info("Starting the GFAS fire radiative power plotting...")

# --------------- Creating plots ---------------
fig, axs = plt.subplots(nrows = nrows, ncols = ncolumns,
                        figsize = (10,8), layout='tight')

sort_gfas_frp = np.argsort(gfas_frp)
gfas_frp_sorted = np.array(gfas_frp)[sort_gfas_frp]
gfas_frp_range = np.logspace(np.log10(gfas_frp_sorted.min()),
                                    np.log10(gfas_frp_sorted.max()),
                                    200)

# ----------------------------------------------------------------------------------------------------

gfas_fwi_sorted = np.array(gfas_fwi)[sort_gfas_frp]
log_gfas_fwi = np.log10(gfas_fwi_sorted)

gfas_fwi_smooth, r_squared_fwi = get_log_linear_stats(gfas_frp_range,
                                                      gfas_frp_sorted,
                                                      gfas_fwi_sorted)

gfas_fwi = axs[0, 0].scatter(
    x=gfas_frp,
    y=gfas_fwi,
    c=mean_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Fire weather index",
)
axs[0,0].plot(gfas_frp_range, gfas_fwi_smooth, "r--", label =
              f"Power trendline (FWI)\nR2={round(r_squared_fwi,3)}")

axs[0,0].set_xscale("log")
axs[0,0].legend(loc=0)
axs[0,0].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[0,0].set_ylabel("FWI Fire weather index", size=fontsize)
start, end = 0, 35
axs[0,0].set_ylim([start,end])
axs[0,0].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,0].set_title("a.", loc="left")

# ----------------------------------------------------------------------------------------------------
gfas_bi_sorted = np.array(gfas_bi)[sort_gfas_frp]

gfas_bi_smooth, r_squared_bi = get_log_linear_stats(gfas_frp_range,
                                                    gfas_frp_sorted,
                                                    gfas_bi_sorted)


gfas_bi = axs[0, 1].scatter(
    x=gfas_frp,
    y=gfas_bi,
    c=mean_values_colour,
    s=10,
    marker = "o",
    alpha=0.2,
    label="Burning index",
)
axs[0,1].plot(gfas_frp_range, gfas_bi_smooth, "r--", label =
              f"Power trendline (BI)\nR2={round(r_squared_bi,3)}")
axs[0,1].set_xscale("log")
axs[0,1].legend(loc=0)
axs[0,1].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[0,1].set_ylabel("NFDRS Burning index", size=fontsize)
start, end = 0, 40
axs[0,1].set_ylim([start,end])
axs[0,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[0,1].set_title("b.", loc="left")

# ----------------------------------------------------------------------------------------------------
gfas_bui_sorted = np.array(gfas_bui)[sort_gfas_frp]

gfas_bui_smooth, r_squared_bui = get_log_linear_stats(gfas_frp_range,
                                                      gfas_frp_sorted,
                                                      gfas_bui_sorted)

gfas_bui = axs[1, 0].scatter(
    x=gfas_frp,
    y=gfas_bui,
    c=mean_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Buildup index",
)
axs[1,0].plot(gfas_frp_range, gfas_bui_smooth, "r--", label =
              f"Power trendline (BUI)\nR2={round(r_squared_bui,3)}")

axs[1,0].set_xscale("log")
axs[1,0].legend(loc=0)
axs[1,0].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[1,0].set_ylabel("FWI Buildup index", size=fontsize)
start, end = 0, 170
axs[1,0].set_ylim([start,end])
axs[1,0].yaxis.set_ticks(np.arange(start,end+1,10))
axs[1,0].set_title("c.", loc="left")

# ----------------------------------------------------------------------------------------------------
gfas_erc_sorted = np.array(gfas_erc)[sort_gfas_frp]

gfas_erc_smooth, r_squared_erc = get_log_linear_stats(gfas_frp_range,
                                                      gfas_frp_sorted,
                                                      gfas_erc_sorted)

gfas_erc = axs[1, 1].scatter(
    x=gfas_frp,
    y=gfas_erc,
    c=mean_values_colour,
    s=10,
    marker = "v",
    alpha=0.2,
    label="Energy release component",
)
axs[1,1].plot(gfas_frp_range, gfas_erc_smooth, "r--", label =
              f"Power trendline (ERC)\nR2={round(r_squared_erc,3)}")

axs[1,1].set_xscale("log")
axs[1,1].legend(loc=0)
axs[1,1].set_xlabel("Fire radiative power (W/m2)", size=fontsize)
axs[1,1].set_ylabel("NFDRS Energy release component", size=fontsize)
start, end = 0, 40
axs[1,1].set_ylim([start,end])
axs[1,1].yaxis.set_ticks(np.arange(start,end+1,5))
axs[1,1].set_title("d.", loc="left")

plt.savefig('/PATH/TO/gfas_frp.png', dpi=500)
#plt.show()
