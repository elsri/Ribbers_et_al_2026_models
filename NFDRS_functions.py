import numpy as np
from numba import njit, prange

@njit(fastmath=True, boundscheck=True)
def FrozenPregreen(
    day,
    fourth_previous_day_temperature_mean,
    third_previous_day_temperature_mean,
    second_previous_day_temperature_mean,
    previous_day_temperature_mean,
    previous_day_frozen,
    temperature_mean,
):
    if day <= 5:
        frozen = 1
    else:
        if previous_day_frozen == 1:
            if (
                fourth_previous_day_temperature_mean > 41
                and third_previous_day_temperature_mean > 41
                and second_previous_day_temperature_mean > 41
                and previous_day_temperature_mean > 41
                and temperature_mean > 41
            ):
                frozen = 0
            else:
                frozen = previous_day_frozen
        else:
            frozen = previous_day_frozen
    return frozen

@njit(fastmath=True, boundscheck=True)
def HerbaceousIndependentVariablePregreen(
    dead_thousand_hour_fuel_moisture,
):
    herbaceous_independent_variable = dead_thousand_hour_fuel_moisture
    return herbaceous_independent_variable

@njit(fastmath=True, boundscheck=True)
def MoistureContentHerbaceousPregreen(
    dead_one_hour_fuel_moisture,
):
    moisture_content_herbaceous = dead_one_hour_fuel_moisture
    if moisture_content_herbaceous < 0:
        moisture_content_herbaceous = 0
    return moisture_content_herbaceous

@njit(fastmath=True, boundscheck=True)
def MoistureContentWoodyPregreen(
    moisture_content_wood_pregreen_stage,
):
    moisture_content_woody = moisture_content_wood_pregreen_stage
    return moisture_content_woody

@njit(fastmath=True, boundscheck=True)
def OneHourFuelLoadingPregreen(
    fuel_model_one_hour_fuel_loading,
    fuel_model_herbaceous_fuel_loading,
):
    one_hour_fuel_loading = (
        fuel_model_one_hour_fuel_loading + fuel_model_herbaceous_fuel_loading
    )
    return one_hour_fuel_loading

@njit(fastmath=True, boundscheck=True)
def HerbaceousFuelLoadingPregreen(
    fuel_model_herbaceous_fuel_loading,
):
    herbaceous_fuel_loading = fuel_model_herbaceous_fuel_loading
    return herbaceous_fuel_loading

@njit(fastmath=True, boundscheck=True)
def HerbaceousIndependentVariableGreenup(
    dead_thousand_hour_fuel_moisture,
    day,
    day_change,
    previous_day_dead_thousand_hour_fuel_moisture,
    previous_day_herbaceous_independent_variable,
    temperature_max,
    temperature_min,
):
    if day == day_change:
        herbaceous_independent_variable = dead_thousand_hour_fuel_moisture
    else:
        difference = (
            dead_thousand_hour_fuel_moisture
            - previous_day_dead_thousand_hour_fuel_moisture
        )
        if dead_thousand_hour_fuel_moisture > 25 or difference <= 0:
            wetting_factor = 1
        elif (
            dead_thousand_hour_fuel_moisture < 26
            and dead_thousand_hour_fuel_moisture > 9
        ):
            wetting_factor = 0.0333 * dead_thousand_hour_fuel_moisture + 0.1675
        elif dead_thousand_hour_fuel_moisture < 10:
            wetting_factor = 0.5

        if (temperature_max + temperature_min) / 2 <= 50:
            temperature_factor = 0.6
        else:
            temperature_factor = 1

        herbaceous_independent_variable = (
            previous_day_herbaceous_independent_variable
            + (difference * wetting_factor * temperature_factor)
        )
    return herbaceous_independent_variable

@njit(fastmath=True, boundscheck=True)
def MoistureContentHerbaceousGreenup(
    day,
    day_change,
    climate_class,
    herbaceous_independent_variable,
    herbaceous_greenup_coefficient,
    herbaceous_greenup_constant,
    previous_day_herbaceous_independent_variable,
):
    greenday = day - day_change
    greenup_period_fraction = greenday / (7.0 * climate_class)
    potential_moisture_content_herbaceous = (
        herbaceous_greenup_constant
        + herbaceous_greenup_coefficient * herbaceous_independent_variable
    )

    moisture_content_herbaceous = (
        previous_day_herbaceous_independent_variable
        + (
            potential_moisture_content_herbaceous
            - previous_day_herbaceous_independent_variable
        )
        * greenup_period_fraction
    )

    if moisture_content_herbaceous > 120:
        moisture_content_herbaceous = 120
    return moisture_content_herbaceous

@njit(fastmath=True, boundscheck=True)
def MoistureContentWoodyGreenup(
    day,
    day_change,
    climate_class,
    previous_day_moisture_content_woody,
    moisture_content_wood_pregreen_stage,
    wood_greenup_coefficient,
    wood_greenup_constant,
    dead_thousand_hour_fuel_moisture,
):
    if day == day_change:
        moisture_content_woody = moisture_content_wood_pregreen_stage
    else:
        greenday = day - day_change
        greenup_period_fraction = greenday / (7.0 * climate_class)
        potential_moisture_content_woody = (
            wood_greenup_constant
            + wood_greenup_coefficient * dead_thousand_hour_fuel_moisture
        )

        if previous_day_moisture_content_woody > moisture_content_wood_pregreen_stage:
            moisture_woody_index = previous_day_moisture_content_woody
        elif moisture_content_wood_pregreen_stage > previous_day_moisture_content_woody:
            moisture_woody_index = moisture_content_wood_pregreen_stage
        else:
            moisture_woody_index = previous_day_moisture_content_woody

        moisture_content_woody = (
            moisture_woody_index
            + (potential_moisture_content_woody - moisture_woody_index)
            * greenup_period_fraction
        )
        if moisture_content_woody < moisture_content_wood_pregreen_stage:
            moisture_content_woody = moisture_content_wood_pregreen_stage
        elif moisture_content_woody > 200:
            moisture_content_woody = 200
    return moisture_content_woody

@njit(fastmath=True, boundscheck=True)
def OneHourFuelLoadingGreenup(
    fuel_model_one_hour_fuel_loading,
    moisture_content_herbaceous,
    fuel_model_herbaceous_fuel_loading,
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 1

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    one_hour_fuel_loading = (
        fuel_model_one_hour_fuel_loading + herbaceous_fuel_transferred
    )
    return one_hour_fuel_loading

@njit(fastmath=True, boundscheck=True)
def HerbaceousFuelLoadingGreenup(
    fuel_model_herbaceous_fuel_loading, moisture_content_herbaceous
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 0.9999999

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    herbaceous_fuel_loading = (
        fuel_model_herbaceous_fuel_loading - herbaceous_fuel_transferred
    )
    return herbaceous_fuel_loading

@njit(fastmath=True, boundscheck=True)
def HerbaceousIndependentVariableGreen(
    dead_thousand_hour_fuel_moisture,
    previous_day_dead_thousand_hour_fuel_moisture,
    previous_day_herbaceous_independent_variable,
    temperature_max,
    temperature_min,
):
    difference = (
        dead_thousand_hour_fuel_moisture - previous_day_dead_thousand_hour_fuel_moisture
    )
    if dead_thousand_hour_fuel_moisture > 25 or difference <= 0:
        wetting_factor = 1
    elif dead_thousand_hour_fuel_moisture < 26 and dead_thousand_hour_fuel_moisture > 9:
        wetting_factor = 0.0333 * dead_thousand_hour_fuel_moisture + 0.1675
    elif dead_thousand_hour_fuel_moisture < 10:
        wetting_factor = 0.5

    if (temperature_max + temperature_min) / 2 <= 50:
        temperature_factor = 0.6
    else:
        temperature_factor = 1

    herbaceous_independent_variable = previous_day_herbaceous_independent_variable + (
        difference * wetting_factor * temperature_factor
    )
    return herbaceous_independent_variable

@njit(fastmath=True, boundscheck=True)
def MoistureContentHerbaceousGreen(
    herbaceous_independent_variable,
    herbaceous_greenup_coefficient,
    herbaceous_greenup_constant,
):
    moisture_content_herbaceous = (
        herbaceous_greenup_constant
        + herbaceous_greenup_coefficient * herbaceous_independent_variable
    )
    if moisture_content_herbaceous > 250:
        moisture_content_herbaceous = 250
    elif moisture_content_herbaceous < 30:
        moisture_content_herbaceous = 30
    return moisture_content_herbaceous

@njit(fastmath=True, boundscheck=True)
def MoistureContentWoodyGreen(
    moisture_content_wood_pregreen_stage,
    wood_greenup_coefficient,
    wood_greenup_constant,
    dead_thousand_hour_fuel_moisture,
):
    moisture_content_woody = (
        wood_greenup_constant
        + wood_greenup_coefficient * dead_thousand_hour_fuel_moisture
    )
    if moisture_content_woody < moisture_content_wood_pregreen_stage:
        moisture_content_woody = moisture_content_wood_pregreen_stage
    elif moisture_content_woody > 200:
        moisture_content_woody = 200
    return moisture_content_woody

@njit(fastmath=True, boundscheck=True)
def OneHourFuelLoadingGreen(
    fuel_model_one_hour_fuel_loading,
    moisture_content_herbaceous,
    fuel_model_herbaceous_fuel_loading,
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 1

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    one_hour_fuel_loading = (
        fuel_model_one_hour_fuel_loading + herbaceous_fuel_transferred
    )
    return one_hour_fuel_loading

@njit(fastmath=True, boundscheck=True)
def HerbaceousFuelLoadingGreen(
    fuel_model_herbaceous_fuel_loading, moisture_content_herbaceous
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 0.9999999

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    herbaceous_fuel_loading = (
        fuel_model_herbaceous_fuel_loading - herbaceous_fuel_transferred
    )
    return herbaceous_fuel_loading

@njit(fastmath=True, boundscheck=True)
def FrozenTransition(
    fourth_previous_day_temperature_min,
    third_previous_day_temperature_min,
    second_previous_day_temperature_min,
    previous_day_temperature_min,
    previous_day_frozen,
    temperature_min,
):
    if (
        (
            fourth_previous_day_temperature_min >= 26
            and fourth_previous_day_temperature_min <= 32
        )
        and (
            third_previous_day_temperature_min >= 26
            and third_previous_day_temperature_min <= 32
        )
        and (
            second_previous_day_temperature_min >= 26
            and second_previous_day_temperature_min <= 32
        )
        and (previous_day_temperature_min >= 26 and previous_day_temperature_min <= 32)
        and (temperature_min >= 26 and temperature_min <= 32)
        or temperature_min <= 25
    ):
        frozen = 1
    else:
        frozen = previous_day_frozen
    return frozen

@njit(fastmath=True, boundscheck=True)
def HerbaceousIndependentVariableTransition(
    dead_thousand_hour_fuel_moisture,
    previous_day_dead_thousand_hour_fuel_moisture,
    previous_day_herbaceous_independent_variable,
    temperature_max,
    temperature_min,
):
    difference = (
        dead_thousand_hour_fuel_moisture - previous_day_dead_thousand_hour_fuel_moisture
    )
    if dead_thousand_hour_fuel_moisture > 25 or difference <= 0:
        wetting_factor = 1
    elif dead_thousand_hour_fuel_moisture < 26 and dead_thousand_hour_fuel_moisture > 9:
        wetting_factor = 0.0333 * dead_thousand_hour_fuel_moisture + 0.1675
    elif dead_thousand_hour_fuel_moisture < 10:
        wetting_factor = 0.5

    if (temperature_max + temperature_min) / 2 <= 50:
        temperature_factor = 0.6
    else:
        temperature_factor = 1

    herbaceous_independent_variable = previous_day_herbaceous_independent_variable + (
        difference * wetting_factor * temperature_factor
    )
    return herbaceous_independent_variable

@njit(fastmath=True, boundscheck=True)
def MoistureContentHerbaceousTransition(
    herbaceous_independent_variable,
    perennial_herbaceous_transition_coefficient,
    perennial_herbaceous_transition_constant,
):
    moisture_content_herbaceous = (
        perennial_herbaceous_transition_constant
        + perennial_herbaceous_transition_coefficient * herbaceous_independent_variable
    )
    if moisture_content_herbaceous < 30:
        moisture_content_herbaceous = 30
    elif moisture_content_herbaceous > 150:
        moisture_content_herbaceous = 150
    return moisture_content_herbaceous

@njit(fastmath=True, boundscheck=True)
def MoistureContentWoodyTransition(
    moisture_content_wood_pregreen_stage,
    wood_greenup_coefficient,
    wood_greenup_constant,
    dead_thousand_hour_fuel_moisture,
    frozen,
):
    if frozen == 0:
        moisture_content_woody = (
            wood_greenup_constant
            + wood_greenup_coefficient * dead_thousand_hour_fuel_moisture
        )
        if moisture_content_woody < moisture_content_wood_pregreen_stage:
            moisture_content_woody = moisture_content_wood_pregreen_stage
        elif moisture_content_woody > 200:
            moisture_content_woody = 200
    else:
        moisture_content_woody = moisture_content_wood_pregreen_stage
    return moisture_content_woody

@njit(fastmath=True, boundscheck=True)
def OneHourFuelLoadingTransition(
    fuel_model_one_hour_fuel_loading,
    moisture_content_herbaceous,
    fuel_model_herbaceous_fuel_loading,
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 1

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    one_hour_fuel_loading = (
        fuel_model_one_hour_fuel_loading + herbaceous_fuel_transferred
    )
    return one_hour_fuel_loading

@njit(fastmath=True, boundscheck=True)
def HerbaceousFuelLoadingTransition(
    fuel_model_herbaceous_fuel_loading, moisture_content_herbaceous
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 0.9999999

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    herbaceous_fuel_loading = (
        fuel_model_herbaceous_fuel_loading - herbaceous_fuel_transferred
    )
    return herbaceous_fuel_loading

@njit(fastmath=True, boundscheck=True) 
def FrozenCured(
    fourth_previous_day_temperature_min,
    third_previous_day_temperature_min,
    second_previous_day_temperature_min,
    previous_day_temperature_min,
    previous_day_frozen,
    temperature_min,
):
    if previous_day_frozen == 0:
        if (
            (
                fourth_previous_day_temperature_min >= 26
                and fourth_previous_day_temperature_min <= 32
            )
            and (
                third_previous_day_temperature_min >= 26
                and third_previous_day_temperature_min <= 32
            )
            and (
                second_previous_day_temperature_min >= 26
                and second_previous_day_temperature_min <= 32
            )
            and (
                previous_day_temperature_min >= 26
                and previous_day_temperature_min <= 32
            )
            and (temperature_min >= 26 and temperature_min <= 32)
            or temperature_min <= 25
        ):
            frozen = 1
        else:
            frozen = previous_day_frozen
    else:
        frozen = previous_day_frozen
    return frozen

@njit(fastmath=True, boundscheck=True)
def HerbaceousIndependentVariableCured(
    dead_thousand_hour_fuel_moisture,
    previous_day_dead_thousand_hour_fuel_moisture,
    previous_day_herbaceous_independent_variable,
    temperature_max,
    temperature_min,
):
    difference = (
        dead_thousand_hour_fuel_moisture - previous_day_dead_thousand_hour_fuel_moisture
    )
    if dead_thousand_hour_fuel_moisture > 25 or difference <= 0:
        wetting_factor = 1
    elif dead_thousand_hour_fuel_moisture < 26 and dead_thousand_hour_fuel_moisture > 9:
        wetting_factor = 0.0333 * dead_thousand_hour_fuel_moisture + 0.1675
    elif dead_thousand_hour_fuel_moisture < 10:
        wetting_factor = 0.5

    if (temperature_max + temperature_min) / 2 <= 50:
        temperature_factor = 0.6
    else:
        temperature_factor = 1

    herbaceous_independent_variable = previous_day_herbaceous_independent_variable + (
        difference * wetting_factor * temperature_factor
    )
    return herbaceous_independent_variable

@njit(fastmath=True, boundscheck=True) 
def MoistureContentHerbaceousCured(
    dead_one_hour_fuel_moisture,
    herbaceous_independent_variable,
    perennial_herbaceous_transition_coefficient,
    perennial_herbaceous_transition_constant,
    frozen,
):
    if frozen == 0:
        moisture_content_herbaceous = (
            perennial_herbaceous_transition_constant
            + perennial_herbaceous_transition_coefficient
            * herbaceous_independent_variable
        )
        if moisture_content_herbaceous < 30:
            moisture_content_herbaceous = 30
        elif moisture_content_herbaceous > 150:
            moisture_content_herbaceous = 150
    else:
        moisture_content_herbaceous = dead_one_hour_fuel_moisture
    return moisture_content_herbaceous

@njit(fastmath=True, boundscheck=True) 
def MoistureContentWoodyCured(
    moisture_content_wood_pregreen_stage,
    wood_greenup_coefficient,
    wood_greenup_constant,
    dead_thousand_hour_fuel_moisture,
    frozen,
):
    if frozen == 0:
        moisture_content_woody = (
            wood_greenup_constant
            + wood_greenup_coefficient * dead_thousand_hour_fuel_moisture
        )
        if moisture_content_woody < moisture_content_wood_pregreen_stage:
            moisture_content_woody = moisture_content_wood_pregreen_stage
        elif moisture_content_woody > 200:
            moisture_content_woody = 200
    else:
        moisture_content_woody = moisture_content_wood_pregreen_stage
    return moisture_content_woody

@njit(fastmath=True, boundscheck=True)  
def HerbaceousFuelLoadingCured(
    fuel_model_herbaceous_fuel_loading, moisture_content_herbaceous
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 0.9999999

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    herbaceous_fuel_loading = (
        fuel_model_herbaceous_fuel_loading - herbaceous_fuel_transferred
    )
    return herbaceous_fuel_loading

@njit(fastmath=True, boundscheck=True)  
def OneHourFuelLoadingCured(
    fuel_model_one_hour_fuel_loading,
    moisture_content_herbaceous,
    fuel_model_herbaceous_fuel_loading,
):
    fraction_transferred_herbaceous_fuel_loading = (
        1.33 - 0.0111 * moisture_content_herbaceous
    )
    if fraction_transferred_herbaceous_fuel_loading < 0:
        fraction_transferred_herbaceous_fuel_loading = 0
    elif fraction_transferred_herbaceous_fuel_loading > 1:
        fraction_transferred_herbaceous_fuel_loading = 1

    herbaceous_fuel_transferred = (
        fraction_transferred_herbaceous_fuel_loading
        * fuel_model_herbaceous_fuel_loading
    )
    one_hour_fuel_loading = (
        fuel_model_one_hour_fuel_loading + herbaceous_fuel_transferred
    )
    return one_hour_fuel_loading

@njit(fastmath=True, boundscheck=True)
def DayAndGridLoops(
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
    boundary_condition_seven_day_input):

    for i in prange(len_i): ### The grid loop
        for day in range(days_in_year): ### The day loop, using prange for parallel computing

            ########################################
            index_x = all_indexes_lon_np[i]
            index_y = all_indexes_lat_np[i]
        
            latitude = latitudes[index_y, index_x]
            slope = slope_yx[index_y,index_x]
        
            # Extract parameters from the array para_yx.
            # The array para_yx is computed by the Para function.
            #
            fuel_model_one_hour_fuel_loading      = para_yx[ 0,index_y,index_x]
            surface_area_volume_one_hour          = para_yx[ 1,index_y,index_x]
            fuel_model_ten_hour_fuel_loading      = para_yx[ 2,index_y,index_x]
            surface_area_volume_ten_hour          = para_yx[ 3,index_y,index_x]
            fuel_model_hundred_hour_fuel_loading  = para_yx[ 4,index_y,index_x]
            surface_area_volume_hundred_hour      = para_yx[ 5,index_y,index_x]
            fuel_model_thousand_hour_fuel_loading = para_yx[ 6,index_y,index_x]
            surface_area_volume_thousand_hour     = para_yx[ 7,index_y,index_x]
            fuel_model_herbaceous_fuel_loading    = para_yx[ 8,index_y,index_x]
            surface_area_volume_herbaceous        = para_yx[ 9,index_y,index_x]
            fuel_model_woody_fuel_loading         = para_yx[10,index_y,index_x]
            surface_area_volume_woody             = para_yx[11,index_y,index_x]
            fuel_bed_depth                        = para_yx[13,index_y,index_x]
            dead_fuel_extinction_moisture         = para_yx[14,index_y,index_x]
            fuel_heat_combustion                  = para_yx[15,index_y,index_x]
            specified_spread_component            = para_yx[16,index_y,index_x]
            #
            # Adjust parameters based on LU_indices and day.
            # The arrays LU20 and LU21 are computed by the Para function.
            #
            if LU_indices[index_y, index_x] == 20:  # Deciduous and evergreen broadleaf forest            
                if day <= 91 or day >= 274:  # Winter litter
                    fuel_model_one_hour_fuel_loading      = para_LU20_d91[0]
                    surface_area_volume_one_hour          = para_LU20_d91[1]
                    fuel_model_ten_hour_fuel_loading      = para_LU20_d91[2]
                    surface_area_volume_ten_hour          = para_LU20_d91[3]
                    fuel_model_hundred_hour_fuel_loading  = para_LU20_d91[4]
                    surface_area_volume_hundred_hour      = para_LU20_d91[5]
                    fuel_model_thousand_hour_fuel_loading = para_LU20_d91[6]
                    surface_area_volume_thousand_hour     = para_LU20_d91[7]
                    fuel_model_woody_fuel_loading         = para_LU20_d91[10]
                    fuel_bed_depth                        = para_LU20_d91[13]
                    specified_spread_component            = para_LU20_d91[16]
                else:  # Summer litter              
                    fuel_model_one_hour_fuel_loading      = para_LU20_else[0]
                    surface_area_volume_one_hour          = para_LU20_else[1]
                    fuel_model_ten_hour_fuel_loading      = para_LU20_else[2]
                    surface_area_volume_ten_hour          = para_LU20_else[3]
                    fuel_model_hundred_hour_fuel_loading  = para_LU20_else[4]
                    surface_area_volume_hundred_hour      = para_LU20_else[5]
                    fuel_model_thousand_hour_fuel_loading = para_LU20_else[6]
                    surface_area_volume_thousand_hour     = para_LU20_else[7]
                    fuel_model_woody_fuel_loading         = para_LU20_else[10]
                    fuel_bed_depth                        = para_LU20_else[13]
                    specified_spread_component            = para_LU20_else[16]
            elif LU_indices[index_y, index_x] == 21:  # Mixed forest
                if day <= 91 or day >= 274:  # Winter litter
                    fuel_model_one_hour_fuel_loading      = para_LU21_d91[0]
                    surface_area_volume_one_hour          = para_LU21_d91[1]
                    fuel_model_ten_hour_fuel_loading      = para_LU21_d91[2]
                    surface_area_volume_ten_hour          = para_LU21_d91[3]
                    fuel_model_hundred_hour_fuel_loading  = para_LU21_d91[4]
                    surface_area_volume_hundred_hour      = para_LU21_d91[5]
                    fuel_model_thousand_hour_fuel_loading = para_LU21_d91[6]
                    surface_area_volume_thousand_hour     = para_LU21_d91[7]
                    fuel_model_woody_fuel_loading         = para_LU21_d91[10]
                    fuel_bed_depth                        = para_LU21_d91[13]
                    dead_fuel_extinction_moisture         = para_LU21_d91[14]
                    fuel_heat_combustion                  = para_LU21_d91[15]
                    specified_spread_component            = para_LU21_d91[16]
                else:  # Summer litter                                             
                    fuel_model_one_hour_fuel_loading      = para_LU21_else[0]
                    surface_area_volume_one_hour          = para_LU21_else[1]
                    fuel_model_ten_hour_fuel_loading      = para_LU21_else[2]
                    surface_area_volume_ten_hour          = para_LU21_else[3]
                    fuel_model_hundred_hour_fuel_loading  = para_LU21_else[4]
                    surface_area_volume_hundred_hour      = para_LU21_else[5]
                    fuel_model_thousand_hour_fuel_loading = para_LU21_else[6]
                    surface_area_volume_thousand_hour     = para_LU21_else[7]
                    fuel_model_woody_fuel_loading         = para_LU21_else[10]
                    fuel_bed_depth                        = para_LU21_else[13]
                    dead_fuel_extinction_moisture         = para_LU21_else[14]
                    fuel_heat_combustion                  = para_LU21_else[15]
                    specified_spread_component            = para_LU21_else[16]

            ##################################################################
            
            temperature_mean = temperature_mean_input[day,index_y, index_x]
            humidities_mean  = relative_humidity_at_2m_input[day,index_y, index_x]
            humidities_min   = humidity_min_input[day,index_y, index_x]
            humidities_max   = humidity_max_input[day,index_y, index_x]
            windspeeds       = wind_speeds_at_10m_input[day,index_y, index_x] * 0.621371192 # km/h to mph

            precipitation_duration = precipitation_input[day,index_y, index_x]
            if precipitation_duration > 8:
                precipitation_duration = 8
                
            temperature_max = temperature_max_input[day,index_y, index_x]
            temperature_min = temperature_min_input[day,index_y, index_x]

            ##########################################################################
            #
            # Extract climate parameters from the Cpara_yx array.
            # The array Cpara_yx is computed by the Cpara function.
            
            climate_class                               = Cpara_yx[ 0,index_y,index_x] 
            herbaceous_greenup_constant                 = Cpara_yx[ 1,index_y,index_x]
            herbaceous_greenup_coefficient              = Cpara_yx[ 2,index_y,index_x]
            perennial_herbaceous_transition_constant    = Cpara_yx[ 5,index_y,index_x]
            perennial_herbaceous_transition_coefficient = Cpara_yx[ 6,index_y,index_x]
            moisture_content_wood_pregreen_stage        = Cpara_yx[ 7,index_y,index_x]
            wood_greenup_constant                       = Cpara_yx[ 8,index_y,index_x]
            wood_greenup_coefficient                    = Cpara_yx[ 9,index_y,index_x]
            greenup_period_days                         = Cpara_yx[12,index_y,index_x]
            
            ##########################################################################

            """
            Dead-Fuel Moisture models
            """

            # logger.info("Starting the dead-fuel moisture model...")

            duration_daylight = DurationDaylight(latitude, day)

            equilibrium_moisture_content_mean = MoistureContentMean(
                humidities_mean, temperature_mean
            )
            if equilibrium_moisture_content_mean < 0:
                equilibrium_moisture_content_mean = 0

            equilibrium_moisture_content_max = MoistureContentMax(
                humidities_max, temperature_min
            )
            equilibrium_moisture_content_min = MoistureContentMin(
                humidities_min, temperature_max
            )

            dead_one_hour_fuel_moisture = DeadOneHourFuelMoisture(
                equilibrium_moisture_content_mean,
                precipitation_duration,
            )
            dead_ten_hour_fuel_moisture = DeadTenHourFuelMoisture(
                equilibrium_moisture_content_mean
            )

            humidities_max = humidity_max_input[day, index_y, index_x]
            humidities_min = humidity_min_input[day, index_y, index_x]

            moisture_content_weighted_average = MoistureContentWeightedAverage(
                duration_daylight,
                equilibrium_moisture_content_max,
                equilibrium_moisture_content_min,
            )

            if year == startyear and day == 0:
                previous_day_hundred_hour_fuel_moisture  = 5.0 + (5.0 * climate_class)
                previous_day_thousand_hour_fuel_moisture = 10.0 + (5.0 * climate_class)
            elif year > startyear and day == 0:
                previous_day_hundred_hour_fuel_moisture = (
                    dead_hundred_hour_fuel_moisture_input[index_y, index_x]
                )
                previous_day_thousand_hour_fuel_moisture = (
                    dead_thousand_hour_fuel_moisture_input[index_y, index_x]
                )
            else:
                previous_day_hundred_hour_fuel_moisture = (
                    dead_hundred_hour_fuel_moisture_output[day - 1, index_y, index_x]
                )
                previous_day_thousand_hour_fuel_moisture = (
                    dead_thousand_hour_fuel_moisture_output[day - 1, index_y, index_x]
                )

            dead_hundred_hour_fuel_moisture = DeadHundredHourFuelMoisture(
                previous_day_hundred_hour_fuel_moisture,
                precipitation_duration,
                moisture_content_weighted_average,
            )
            boundary_condition_weighted_average_thousand = (
                BoundaryConditionWeightedAverageThousandHour(
                    precipitation_duration, moisture_content_weighted_average
                )
            )

            if (day + 1) % 7 == 0 and day >= 6:
                previous_day_boundary_condition_weighted_average_thousand = (
                    boundary_condition_weighted_average_thousand_output[
                        day - 1, index_y, index_x
                    ]
                )
                second_previous_day_boundary_condition_weighted_average_thousand = (
                    boundary_condition_weighted_average_thousand_output[
                        day - 2, index_y, index_x
                    ]
                )
                third_previous_day_boundary_condition_weighted_average_thousand = (
                    boundary_condition_weighted_average_thousand_output[
                        day - 3, index_y, index_x
                    ]
                )
                fourth_previous_day_boundary_condition_weighted_average_thousand = (
                    boundary_condition_weighted_average_thousand_output[
                        day - 4, index_y, index_x
                    ]
                )
                fifth_previous_day_boundary_condition_weighted_average_thousand = (
                    boundary_condition_weighted_average_thousand_output[
                        day - 5, index_y, index_x
                    ]
                )
                sixth_previous_day_boundary_condition_weighted_average_thousand = (
                    boundary_condition_weighted_average_thousand_output[
                        day - 6, index_y, index_x
                    ]
                )
                previous_day_boundary_condition_seven_day = (
                    boundary_condition_seven_day_output[day - 1, index_y, index_x]
                )
            elif year > startyear and day == 0:
                previous_day_boundary_condition_weighted_average_thousand = 0
                second_previous_day_boundary_condition_weighted_average_thousand = 0
                third_previous_day_boundary_condition_weighted_average_thousand = 0
                fourth_previous_day_boundary_condition_weighted_average_thousand = 0
                fifth_previous_day_boundary_condition_weighted_average_thousand = 0
                sixth_previous_day_boundary_condition_weighted_average_thousand = 0
                previous_day_boundary_condition_seven_day = (
                    boundary_condition_seven_day_input[index_y, index_x]
                )
            elif year == startyear and day == 0:
                previous_day_boundary_condition_weighted_average_thousand = 0
                second_previous_day_boundary_condition_weighted_average_thousand = 0
                third_previous_day_boundary_condition_weighted_average_thousand = 0
                fourth_previous_day_boundary_condition_weighted_average_thousand = 0
                fifth_previous_day_boundary_condition_weighted_average_thousand = 0
                sixth_previous_day_boundary_condition_weighted_average_thousand = 0
                previous_day_boundary_condition_seven_day = 0
            else:
                previous_day_boundary_condition_weighted_average_thousand = 0
                second_previous_day_boundary_condition_weighted_average_thousand = 0
                third_previous_day_boundary_condition_weighted_average_thousand = 0
                fourth_previous_day_boundary_condition_weighted_average_thousand = 0
                fifth_previous_day_boundary_condition_weighted_average_thousand = 0
                sixth_previous_day_boundary_condition_weighted_average_thousand = 0
                previous_day_boundary_condition_seven_day = (
                boundary_condition_seven_day_output[day - 1, index_y, index_x]
                )

            boundary_condition_seven_day = BoundaryConditionSevenDayAverageThousandHour(
                boundary_condition_weighted_average_thousand,
                previous_day_boundary_condition_weighted_average_thousand,
                second_previous_day_boundary_condition_weighted_average_thousand,
                third_previous_day_boundary_condition_weighted_average_thousand,
                fourth_previous_day_boundary_condition_weighted_average_thousand,
                fifth_previous_day_boundary_condition_weighted_average_thousand,
                sixth_previous_day_boundary_condition_weighted_average_thousand,
                previous_day_boundary_condition_seven_day,
                day,
                year,
                startyear,
                climate_class,
            )

            dead_thousand_hour_fuel_moisture = DeadThousandHourFuelMoisture(
                previous_day_thousand_hour_fuel_moisture,
                boundary_condition_seven_day,
                day,
            )

            """       
            Live-Fuel Moisture models   
            """

            if day == 0:
                phase_herbaceous = 1
                frozen = 1
                day_change = 0
            else:
                phase_herbaceous = phase_herbaceous_output[day - 1, index_y, index_x]
                frozen = frozen_output[day - 1, index_y, index_x]
                day_change = day_change_output[day - 1, index_y, index_x]

            if phase_herbaceous == 1:
                if day >= 4:
                    fourth_previous_day_temperature_mean = temperature_mean_input[
                        day - 4, index_y, index_x
                    ]
                    third_previous_day_temperature_mean = temperature_mean_input[
                        day - 3, index_y, index_x
                    ]
                    second_previous_day_temperature_mean = temperature_mean_input[
                        day - 2, index_y, index_x
                    ]
                    previous_day_temperature_mean = temperature_mean_input[
                        day - 1, index_y, index_x
                    ]
                    previous_day_frozen = frozen_output[day - 1, index_y, index_x]
                else:
                    fourth_previous_day_temperature_mean = 0
                    third_previous_day_temperature_mean = 0
                    second_previous_day_temperature_mean = 0
                    previous_day_temperature_mean = 0
                    previous_day_frozen = 1

                frozen = FrozenPregreen(
                    day,
                    fourth_previous_day_temperature_mean,
                    third_previous_day_temperature_mean,
                    second_previous_day_temperature_mean,
                    previous_day_temperature_mean,
                    previous_day_frozen,
                    temperature_mean,
                )

                herbaceous_independent_variable = HerbaceousIndependentVariablePregreen(
                    dead_thousand_hour_fuel_moisture,
                )

                moisture_content_herbaceous = MoistureContentHerbaceousPregreen(
                    dead_one_hour_fuel_moisture,
                )

                herbaceous_fuel_loading = HerbaceousFuelLoadingPregreen(
                    fuel_model_herbaceous_fuel_loading,
                )

                moisture_content_woody = MoistureContentWoodyPregreen(
                    moisture_content_wood_pregreen_stage,
                )

                one_hour_fuel_loading = OneHourFuelLoadingPregreen(
                    fuel_model_one_hour_fuel_loading,
                    fuel_model_herbaceous_fuel_loading,
                )

            elif phase_herbaceous == 2:
                if greenup_period_days <= 0 or day == 0:
                    greenup_period_fraction = 0.0
                else:
                    greenup_period_fraction = ((day - 1) - day_change) / greenup_period_days

                if day == 0:
                    frozen = 1
                    previous_day_herbaceous_independent_variable = 0
                    previous_day_dead_thousand_hour_fuel_moisture = 0
                else:
                    frozen = frozen_output[day - 1, index_y, index_x]
                    previous_day_herbaceous_independent_variable = (
                        herbaceous_independent_variable_output[day - 1, index_y, index_x]
                    )
                    previous_day_dead_thousand_hour_fuel_moisture = (
                        dead_thousand_hour_fuel_moisture_output[day - 1, index_y, index_x]
                    )
                

                herbaceous_independent_variable = HerbaceousIndependentVariableGreenup(
                    dead_thousand_hour_fuel_moisture,
                    day,
                    day_change,
                    previous_day_dead_thousand_hour_fuel_moisture,
                    previous_day_herbaceous_independent_variable,
                    temperature_max,
                    temperature_min,
                )

                moisture_content_herbaceous = MoistureContentHerbaceousGreenup(
                    day,
                    day_change,
                    climate_class,
                    herbaceous_independent_variable,
                    herbaceous_greenup_coefficient,
                    herbaceous_greenup_constant,
                    previous_day_herbaceous_independent_variable,
                )

                herbaceous_fuel_loading = HerbaceousFuelLoadingGreenup(
                    fuel_model_herbaceous_fuel_loading, moisture_content_herbaceous
                )

                if day == 0:
                    previous_day_moisture_content_woody = moisture_content_wood_pregreen_stage
                else:
                    previous_day_moisture_content_woody = moisture_content_woody_output[
                        day - 1, index_y, index_x
                    ]

                moisture_content_woody = MoistureContentWoodyGreenup(
                    day,
                    day_change,
                    climate_class,
                    previous_day_moisture_content_woody,
                    moisture_content_wood_pregreen_stage,
                    wood_greenup_coefficient,
                    wood_greenup_constant,
                    dead_thousand_hour_fuel_moisture,
                )

                one_hour_fuel_loading = OneHourFuelLoadingGreenup(
                    fuel_model_one_hour_fuel_loading,
                    moisture_content_herbaceous,
                    fuel_model_herbaceous_fuel_loading,
                )

            elif phase_herbaceous == 3:
                frozen = frozen_output[day, index_y, index_x]

                if day == 0:
                    previous_day_herbaceous_independent_variable = 0
                    previous_day_dead_thousand_hour_fuel_moisture = 0
                else:
                    previous_day_herbaceous_independent_variable = (
                        herbaceous_independent_variable_output[day - 1, index_y, index_x]
                    )
                    previous_day_dead_thousand_hour_fuel_moisture = (
                        dead_thousand_hour_fuel_moisture_output[day - 1, index_y, index_x]
                    )

                herbaceous_independent_variable = HerbaceousIndependentVariableGreen(
                    dead_thousand_hour_fuel_moisture,
                    previous_day_dead_thousand_hour_fuel_moisture,
                    previous_day_herbaceous_independent_variable,
                    temperature_max,
                    temperature_min,
                )

                moisture_content_herbaceous = MoistureContentHerbaceousGreen(
                    herbaceous_independent_variable,
                    herbaceous_greenup_coefficient,
                    herbaceous_greenup_constant,
                )

                herbaceous_fuel_loading = HerbaceousFuelLoadingGreen(
                    fuel_model_herbaceous_fuel_loading,
                    moisture_content_herbaceous,
                )

                moisture_content_woody = MoistureContentWoodyGreen(
                    moisture_content_wood_pregreen_stage,
                    wood_greenup_coefficient,
                    wood_greenup_constant,
                    dead_thousand_hour_fuel_moisture,
                )

                one_hour_fuel_loading = OneHourFuelLoadingGreen(
                    fuel_model_one_hour_fuel_loading,
                    moisture_content_herbaceous,
                    fuel_model_herbaceous_fuel_loading,
                )

            elif phase_herbaceous == 4:
                if day >= 4:
                    fourth_previous_day_temperature_min = temperature_min_input[
                        day - 4, index_y, index_x
                    ]
                    third_previous_day_temperature_min = temperature_min_input[
                        day - 3, index_y, index_x
                    ]
                    second_previous_day_temperature_min = temperature_min_input[
                        day - 2, index_y, index_x
                    ]
                    previous_day_temperature_min = temperature_min_input[
                        day - 1, index_y, index_x
                    ]
                    previous_day_frozen = frozen_output[day - 1, index_y, index_x]
                else:
                    fourth_previous_day_temperature_min = 0
                    third_previous_day_temperature_min = 0
                    second_previous_day_temperature_min = 0
                    previous_day_temperature_min = 0
                    previous_day_frozen = 0                    

                frozen = FrozenTransition(
                    fourth_previous_day_temperature_min,
                    third_previous_day_temperature_min,
                    second_previous_day_temperature_min,
                    previous_day_temperature_min,
                    previous_day_frozen,
                    temperature_min,
                )

                if day == 0:
                    previous_day_herbaceous_independent_variable = 0
                    previous_day_dead_thousand_hour_fuel_moisture = 0
                else:
                    previous_day_herbaceous_independent_variable = (
                        herbaceous_independent_variable_output[day - 1, index_y, index_x]
                    )
                    previous_day_dead_thousand_hour_fuel_moisture = (
                        dead_thousand_hour_fuel_moisture_output[day - 1, index_y, index_x]
                    )

                herbaceous_independent_variable = (
                    HerbaceousIndependentVariableTransition(
                        dead_thousand_hour_fuel_moisture,
                        previous_day_dead_thousand_hour_fuel_moisture,
                        previous_day_herbaceous_independent_variable,
                        temperature_max,
                        temperature_min,
                    )
                )

                moisture_content_herbaceous = MoistureContentHerbaceousTransition(
                    herbaceous_independent_variable,
                    perennial_herbaceous_transition_coefficient,
                    perennial_herbaceous_transition_constant,
                )

                herbaceous_fuel_loading = HerbaceousFuelLoadingTransition(
                    fuel_model_herbaceous_fuel_loading,
                    moisture_content_herbaceous,
                )

                moisture_content_woody = MoistureContentWoodyTransition(
                    moisture_content_wood_pregreen_stage,
                    wood_greenup_coefficient,
                    wood_greenup_constant,
                    dead_thousand_hour_fuel_moisture,
                    frozen,
                )

                one_hour_fuel_loading = OneHourFuelLoadingTransition(
                    fuel_model_one_hour_fuel_loading,
                    moisture_content_herbaceous,
                    fuel_model_herbaceous_fuel_loading,
                )

            elif phase_herbaceous == 5:
                if day >= 4:
                    fourth_previous_day_temperature_min = temperature_min_input[
                        day - 4, index_y, index_x
                    ]
                    third_previous_day_temperature_min = temperature_min_input[
                        day - 3, index_y, index_x
                    ]
                    second_previous_day_temperature_min = temperature_min_input[
                        day - 2, index_y, index_x
                    ]
                    previous_day_temperature_min = temperature_min_input[
                        day - 1, index_y, index_x
                    ]
                    previous_day_frozen = frozen_output[day - 1, index_y, index_x]
                else:
                    fourth_previous_day_temperature_min = 0
                    third_previous_day_temperature_min = 0
                    second_previous_day_temperature_min = 0
                    previous_day_temperature_min = 0
                    previous_day_frozen = 0

                frozen = FrozenCured(
                    fourth_previous_day_temperature_min,
                    third_previous_day_temperature_min,
                    second_previous_day_temperature_min,
                    previous_day_temperature_min,
                    previous_day_frozen,
                    temperature_min,
                )

                if day == 0:
                    previous_day_herbaceous_independent_variable = 0
                    previous_day_dead_thousand_hour_fuel_moisture = 0
                else:
                    previous_day_herbaceous_independent_variable = (
                        herbaceous_independent_variable_output[day - 1, index_y, index_x]
                    )
                    previous_day_dead_thousand_hour_fuel_moisture = (
                        dead_thousand_hour_fuel_moisture_output[day - 1, index_y, index_x]
                    )

                herbaceous_independent_variable = HerbaceousIndependentVariableCured(
                    dead_thousand_hour_fuel_moisture,
                    previous_day_dead_thousand_hour_fuel_moisture,
                    previous_day_herbaceous_independent_variable,
                    temperature_max,
                    temperature_min,
                )

                moisture_content_herbaceous = MoistureContentHerbaceousCured(
                    dead_one_hour_fuel_moisture,
                    herbaceous_independent_variable,
                    perennial_herbaceous_transition_coefficient,
                    perennial_herbaceous_transition_constant,
                    frozen,
                )

                herbaceous_fuel_loading = HerbaceousFuelLoadingCured(
                    fuel_model_herbaceous_fuel_loading, moisture_content_herbaceous
                )

                moisture_content_woody = MoistureContentWoodyCured(
                    moisture_content_wood_pregreen_stage,
                    wood_greenup_coefficient,
                    wood_greenup_constant,
                    dead_thousand_hour_fuel_moisture,
                    frozen,
                )

                one_hour_fuel_loading = OneHourFuelLoadingCured(
                    fuel_model_one_hour_fuel_loading,
                    moisture_content_herbaceous,
                    fuel_model_herbaceous_fuel_loading,
                )

            """
            Fires
            """
            if frozen == 1 or precipitation_duration > 0:
                spread_component = 0
                energy_release_component = 0
                burning_index = 0
                category = 0
            else:
                """
                Preliminary calculations
                """
                
                STD_STL_factor = 1-0.0555
                
                # This constant is passed on, during the call to this function.
                net_one_hour_fuel_loading = one_hour_fuel_loading * (
                    STD_STL_factor
                )  # STD and STL set to 0.0555 in NFDR
                net_ten_hour_fuel_loading = fuel_model_ten_hour_fuel_loading * (
                    STD_STL_factor
                )  # STD and STL set to 0.0555 in NFDR
                net_hundred_hour_fuel_loading = fuel_model_hundred_hour_fuel_loading * (
                    STD_STL_factor
                )  # STD and STL set to 0.0555 in NFDR
                net_thousand_hour_fuel_loading = (
                    fuel_model_thousand_hour_fuel_loading * (STD_STL_factor)
                )  # STD and STL set to 0.0555 in NFDR
                net_herbaceous_fuel_loading = herbaceous_fuel_loading * (
                    STD_STL_factor
                )  # STD and STL set to 0.0555 in NFDR
                net_woody_fuel_loading = fuel_model_woody_fuel_loading * (
                    STD_STL_factor
                )  # STD and STL set to 0.0555 in NFDR

                total_dead_fuel_loading = (
                    one_hour_fuel_loading
                    + fuel_model_ten_hour_fuel_loading
                    + fuel_model_hundred_hour_fuel_loading
                    + fuel_model_thousand_hour_fuel_loading
                )
                total_live_fuel_loading = (
                    herbaceous_fuel_loading + fuel_model_woody_fuel_loading
                )
                total_fuel_loading = total_dead_fuel_loading + total_live_fuel_loading

                if total_fuel_loading <= 0:
                    spread_component = 0
                    energy_release_component = 0
                    burning_index = 0
                else:
                    bulk_density_fuel_bed = 0.0
                    if fuel_bed_depth > 0:
                        bulk_density_fuel_bed = (
                            total_fuel_loading - fuel_model_thousand_hour_fuel_loading
                        ) / fuel_bed_depth

                    fuel_particle_density = 32  # assumed in NFDR
                    
                    if total_fuel_loading > 0:
                        weighted_fuel_density = (
                            (total_live_fuel_loading * fuel_particle_density)
                            + (total_dead_fuel_loading * fuel_particle_density)
                        ) / total_fuel_loading

                    if weighted_fuel_density <= 0:
                        packing_ratio = 0.0
                    else:
                        packing_ratio = bulk_density_fuel_bed / weighted_fuel_density

                    packing_ratio = max(packing_ratio, 1e-4)

                    mineral_damping_coefficient = (0.174 * 0.01 ** (-0.19))
                    # 0.01 assumed as fraction of dead or live fuels made up of silica-free, noncombustible minerals

                    heating_number_one_hour = 0.0
                    if surface_area_volume_one_hour > 0:
                        heating_number_one_hour = net_one_hour_fuel_loading * np.exp(
                            -138.0 / surface_area_volume_one_hour
                        )
                    
                    heating_number_ten_hour = 0.0
                    if surface_area_volume_ten_hour > 0:
                        heating_number_ten_hour = net_ten_hour_fuel_loading * np.exp(
                            -138.0 / surface_area_volume_ten_hour
                        )
                    
                    heating_number_hundred_hour = 0.0
                    if surface_area_volume_hundred_hour > 0:
                        heating_number_hundred_hour = (
                            net_hundred_hour_fuel_loading
                            * np.exp(-138.0 / surface_area_volume_hundred_hour)
                        )

                    heating_number_herbaceous = 0.0
                    if surface_area_volume_herbaceous > 0:
                        heating_number_herbaceous = net_herbaceous_fuel_loading * np.exp(
                            -500.0 / surface_area_volume_herbaceous
                        )

                    heating_number_woody = 0.0
                    if surface_area_volume_woody > 0:
                        heating_number_woody = net_woody_fuel_loading * np.exp(
                            -500.0 / surface_area_volume_woody
                        )

                    ratio_heating_numbers = 0.0
                    if heating_number_herbaceous > 0 or heating_number_woody > 0:
                        ratio_heating_numbers = (
                            heating_number_one_hour
                            + heating_number_ten_hour
                            + heating_number_hundred_hour
                        ) / (heating_number_herbaceous + heating_number_woody)

                    surface_area_weighted_characteristic_surface_area_to_volume_ratio = (
                        SurfaceWeightedCharacteristicAreaVolumeRatio(
                            fuel_model_woody_fuel_loading,
                            surface_area_volume_one_hour,
                            surface_area_volume_hundred_hour,
                            surface_area_volume_ten_hour,
                            surface_area_volume_herbaceous,
                            surface_area_volume_woody,
                            fuel_particle_density,
                            one_hour_fuel_loading,
                            fuel_model_ten_hour_fuel_loading,
                            fuel_model_hundred_hour_fuel_loading,
                            herbaceous_fuel_loading,
                        )
                    )
                    """
                    Spread component
                    """ 
                    spread_component = SpreadComponent(
                        net_one_hour_fuel_loading,
                        net_ten_hour_fuel_loading,
                        net_hundred_hour_fuel_loading,
                        net_herbaceous_fuel_loading,
                        net_woody_fuel_loading,
                        one_hour_fuel_loading,
                        fuel_model_ten_hour_fuel_loading,
                        fuel_model_hundred_hour_fuel_loading,
                        herbaceous_fuel_loading,
                        fuel_model_woody_fuel_loading,
                        fuel_particle_density,
                        surface_area_volume_one_hour,
                        surface_area_volume_hundred_hour,
                        surface_area_volume_ten_hour,
                        surface_area_volume_herbaceous,
                        surface_area_volume_woody,
                        packing_ratio,
                        dead_one_hour_fuel_moisture,
                        dead_ten_hour_fuel_moisture,
                        dead_hundred_hour_fuel_moisture,
                        heating_number_one_hour,
                        heating_number_ten_hour,
                        heating_number_hundred_hour,
                        ratio_heating_numbers,
                        moisture_content_herbaceous,
                        moisture_content_woody,
                        dead_fuel_extinction_moisture,
                        slope,
                        fuel_heat_combustion,
                        mineral_damping_coefficient,
                        windspeeds,
                        bulk_density_fuel_bed,
                        surface_area_weighted_characteristic_surface_area_to_volume_ratio,
                    )
                    """
                    Energy Release Component
                    """
                    energy_release_component = EnergyReleaseComponent(
                        one_hour_fuel_loading,
                        fuel_model_ten_hour_fuel_loading,
                        fuel_model_hundred_hour_fuel_loading,
                        fuel_model_thousand_hour_fuel_loading,
                        total_dead_fuel_loading,
                        total_live_fuel_loading,
                        total_fuel_loading,
                        net_woody_fuel_loading,
                        net_herbaceous_fuel_loading,
                        packing_ratio,
                        surface_area_volume_one_hour,
                        surface_area_volume_ten_hour,
                        surface_area_volume_hundred_hour,
                        surface_area_volume_thousand_hour,
                        surface_area_volume_woody,
                        surface_area_volume_herbaceous,
                        dead_fuel_extinction_moisture,
                        heating_number_one_hour,
                        heating_number_ten_hour,
                        heating_number_hundred_hour,
                        ratio_heating_numbers,
                        fuel_heat_combustion,
                        mineral_damping_coefficient,
                        moisture_content_herbaceous,
                        moisture_content_woody,
                        dead_one_hour_fuel_moisture,
                        dead_ten_hour_fuel_moisture,
                        dead_hundred_hour_fuel_moisture,
                        dead_thousand_hour_fuel_moisture,
                        surface_area_weighted_characteristic_surface_area_to_volume_ratio,
                    )
                    if spread_component > specified_spread_component:
                        spread_component = specified_spread_component
            
                    """
                    Burning Index
                    """
                    burning_index = BurningIndex(spread_component, energy_release_component)

                    if burning_index < 30:
                        category = 1
                    elif burning_index >= 30 and burning_index < 40:
                        category = 2
                    elif burning_index >= 40 and burning_index < 60:
                        category = 3
                    elif burning_index >= 60 and burning_index < 80:
                        category = 4
                    elif burning_index >= 80 and burning_index < 90:
                        category = 5
                    else:
                        category = 6

            if (
                day > 0
                and phase_herbaceous == 1
                and moisture_content_herbaceous > 30
                and frozen == 0
                or phase_herbaceous == 1
                and day > 220
            ):
                phase_herbaceous = phase_herbaceous + 1
                day_change = day
            elif phase_herbaceous == 2 and greenup_period_fraction >= 1:
                if moisture_content_herbaceous >= 120:
                    phase_herbaceous = phase_herbaceous + 1
                else:
                    phase_herbaceous = phase_herbaceous + 2
                day_change = day
            elif (
                phase_herbaceous == 3
                and moisture_content_herbaceous < 120
                and day > 173
                or phase_herbaceous == 3
                and latitude > 68
                and duration_daylight < 21
                and temperature_mean_input[day, index_y, index_x] < 41 #41F = 5C
                and temperature_mean_input[day - 1, index_y, index_x] < 41
                and temperature_mean_input[day - 2, index_y, index_x] < 41
                and temperature_mean_input[day - 3, index_y, index_x] < 41
                and temperature_mean_input[day - 4, index_y, index_x] < 41
                and day > 173
                #and temperature_mean <= 50 #50F = 10C
                or phase_herbaceous == 3
                and latitude <= 68
                and latitude > 63
                and duration_daylight < 19
                and temperature_mean_input[day, index_y, index_x] < 41 #41F = 5C
                and temperature_mean_input[day - 1, index_y, index_x] < 41
                and temperature_mean_input[day - 2, index_y, index_x] < 41
                and temperature_mean_input[day - 3, index_y, index_x] < 41
                and temperature_mean_input[day - 4, index_y, index_x] < 41
                and day > 173
                # and temperature_mean <= 50
                or phase_herbaceous == 3
                and latitude <= 63
                and duration_daylight < 17
                and temperature_mean_input[day, index_y, index_x] < 41 #41F = 5C
                and temperature_mean_input[day - 1, index_y, index_x] < 41
                and temperature_mean_input[day - 2, index_y, index_x] < 41
                and temperature_mean_input[day - 3, index_y, index_x] < 41
                and temperature_mean_input[day - 4, index_y, index_x] < 41
                and day > 173
                # and temperature_mean <= 50
            ):
                phase_herbaceous = phase_herbaceous + 1
                day_change = day
            elif (
                phase_herbaceous == 4
                and frozen == 1
                or phase_herbaceous == 4
                and moisture_content_herbaceous <= 30
                or phase_herbaceous == 4
                and day > 360
            ):
                phase_herbaceous = phase_herbaceous + 1
                day_change = day
            else:
                phase_herbaceous = phase_herbaceous

            phase_herbaceous_output[day, index_y, index_x] = phase_herbaceous
            frozen_output[day, index_y, index_x] = frozen
            climate_class_output[day, index_y, index_x] = climate_class
            index_y_output[day, index_y, index_x] = index_y
            index_x_output[day, index_y, index_x] = index_x
            day_output[day, index_y, index_x] = day
            day_change_output[day, index_y, index_x] = day_change
            dead_one_hour_fuel_moisture_output[day, index_y, index_x] = (
                dead_one_hour_fuel_moisture
            )
            dead_ten_hour_fuel_moisture_output[day, index_y, index_x] = (
                dead_ten_hour_fuel_moisture
            )
            dead_hundred_hour_fuel_moisture_output[day, index_y, index_x] = (
                dead_hundred_hour_fuel_moisture
            )
            dead_thousand_hour_fuel_moisture_output[day, index_y, index_x] = (
                dead_thousand_hour_fuel_moisture
            )
            boundary_condition_weighted_average_thousand_output[day, index_y, index_x] = (
                boundary_condition_weighted_average_thousand
            )
            boundary_condition_seven_day_output[day, index_y, index_x] = (
                boundary_condition_seven_day
            )
            herbaceous_independent_variable_output[day, index_y, index_x] = (
                herbaceous_independent_variable
            )
            moisture_content_herbaceous_output[day, index_y, index_x] = (
                moisture_content_herbaceous
            )
            moisture_content_woody_output[day, index_y, index_x] = (
                moisture_content_woody
            )
            spread_component_output[day, index_y, index_x] = spread_component
            energy_release_component_output[day, index_y, index_x] = (
                energy_release_component
            )
            burning_index_output[day, index_y, index_x] = burning_index
            burning_category_output[day, index_y, index_x] = category
            
            duration_daylight_output[day, index_y, index_x] = duration_daylight

    ##################### end of the day and grid loops
    return (duration_daylight_output,
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
            burning_category_output)

@njit(fastmath=True, boundscheck=True)
def Para(
    len_i, all_indexes_lon_np, all_indexes_lat_np, LU_indices,
    number_of_latitudinal_degrees, number_of_longitudinal_degrees):
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
    #
    # Parameter values are calculated using numbers.
    # It is important to perform these calculations only once, outside the loops.
    # Parameter values that are single numbers are included here in case they are
    # later replaced by calculations using numbers.
    #
    # The values depends on the LU_indices.
    #
    #############################################################################
    #if LU_indices[index_y, index_x] == 13:  # Tundra
    
    para_LU13     = np.empty((17))
    para_LU13[0]  = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU13[1]  = 1500
    para_LU13[2]  = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU13[3]  = 109
    para_LU13[4]  = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU13[5]  = 30
    para_LU13[6]  = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU13[7]  = 8
    para_LU13[8]  = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU13[9]  = 1500
    para_LU13[10] = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU13[11] = 1200
    para_LU13[12] = 1.5 #1988 addition
    para_LU13[13] = 0.4
    para_LU13[14] = 25
    para_LU13[15] = 8000
    para_LU13[16] = 17
    
    ######################################################
    #elif LU_indices[index_y, index_x] == 17:  # Grassland
    
    para_LU17     = np.empty((17))
    para_LU17[0]  = (0.25 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU17[1]  = 2000
    para_LU17[2]  = (1.5 * 0.0459137)  # Tons per acre to pounds per square foot
    para_LU17[3]  = 109
    para_LU17[4]  = (0 * 0.0459137)    # Tons per acre to pounds per square foot
    para_LU17[5]  = 0
    para_LU17[6]  = (0 * 0.0459137)    # Tons per acre to pounds per square foot
    para_LU17[7]  = 0
    para_LU17[8]  = (0.5 * 0.0459137)  # Tons per acre to pounds per square foot             
    para_LU17[9]  = 2000
    para_LU17[10] = (0 * 0.0459137)    # Tons per acre to pounds per square foot             
    para_LU17[11] = 0
    para_LU17[12] = 0.25  #1988 addition             
    para_LU17[13] = 1
    para_LU17[14] = 15
    para_LU17[15] = 8000
    para_LU17[16] = 178

    ######################################################################################
    #elif LU_indices[index_y, index_x] == 19:  # Deciduous and evergreen needleleaf forest

    para_LU19     = np.empty((17))
    para_LU19[0]  = ((1 + 2) / 2) * 0.0459137     # Tons per acre to pounds per square foot
    para_LU19[1]  = (1750 + 1500) / 2
    para_LU19[2]  = ((2.5 + 0.5) / 2) * 0.0459137  # Tons per acre to pounds per square foot
    para_LU19[3]  = (109 + 109) / 2
    para_LU19[4]  = ((0.5 + 2) / 2) * 0.0459137  # Tons per acre to pounds per square foot
    para_LU19[5]  = (30 + 30) / 2
    para_LU19[6]  = ((1 + 0) / 2) * 0.0459137  # Tons per acre to pounds per square foot
    para_LU19[7]  = (0 + 8) / 2
    para_LU19[8]  = ((0 + 0.5) / 2) * 0.0459137 # Tons per acre to pounds per square foot
    para_LU19[9]  = (1500 + 2000) / 2
    para_LU19[10] = ((3 + 0.5) / 2) * 0.0459137  # Tons per acre to pounds per square foot
    para_LU19[11] = (1500 + 1200) / 2
    para_LU19[12] = ((3.5 + 2) / 2) * 0.0459137 #Tons per acre to pounds per square foot #1988 addition
    para_LU19[13] = (0.4 + 3) / 2
    para_LU19[14] = (30 + 25) / 2
    para_LU19[15] = (8000 + 8000) / 2
    para_LU19[16] = (14 + 59) / 2

    #####################################################################################
    #elif LU_indices[index_y, index_x] == 20:  # Deciduous and evergreen broadleaf forest
    #    if day <= 91 or day >= 274:  # Winter litter

    para_LU20_d91     = np.empty((17))
    para_LU20_d91[0]  = (1.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU20_d91[1]  = 2000
    para_LU20_d91[2]  = (0.5 * 0.0459137)  # Tons per acre to pounds per square foot
    para_LU20_d91[3]  = 109
    para_LU20_d91[4]  = (0.25 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU20_d91[5]  = 0
    para_LU20_d91[6]  = (0 * 0.0459137)    # Tons per acre to pounds per square foot
    para_LU20_d91[7]  = 0
    para_LU20_d91[10] = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU20_d91[12] = (1.5 * 0.0459137) # Tons per acre to pounds per square foot #1988 addition
    para_LU20_d91[13] = 0.4
    para_LU20_d91[16] = 25
    
    #    else: # Summer litter
    para_LU20_else     = np.empty((17))
    para_LU20_else[0]  = (0.5 * 0.0459137)  # Tons per acre to pounds per square foot
    para_LU20_else[1]  = 1500
    para_LU20_else[2]  = (0.5 * 0.0459137)  # Tons per acre to pounds per square foot
    para_LU20_else[3]  = 109
    para_LU20_else[4]  = (0.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU20_else[5]  = 30
    para_LU20_else[6]  = (0 * 0.0459137)   # Tons per acre to pounds per square foot
    para_LU20_else[7]  = 0
    para_LU20_else[10] = (0.5 * 0.0459137) # Tons per acre to pounds per square foot #1988 adaptation
    para_LU20_else[12] = (0.5 * 0.0459137) #Tons per acre to pounds per square foot
    para_LU20_else[13] = 0.25
    para_LU20_else[16] = 6

    #    LU20:
    para_LU20     = np.empty((17))
    para_LU20[14] = 25
    para_LU20[15] = 8000
    para_LU20[8]  = (0.5 * 0.0459137)  # Tons per acre to pounds per square foot
    para_LU20[9]  = 2000
    para_LU20[11] = 1500

    #########################################################
    #elif LU_indices[index_y, index_x] == 21:  # Mixed forest
    #    if day <= 91 or day >= 274:  # Winter litter    
    #
    para_LU21_d91     = np.empty((17))
    para_LU21_d91[0]  = ((1.5 + 1 + 2) / 3) * 0.0459137   # Tons per acre to pounds per square foot      
    para_LU21_d91[1]  = (2000 + 1750 + 1500) / 3
    para_LU21_d91[2]  = ((0.5 + 2.5 + 0.5) / 3) * 0.0459137 # Tons per acre to pounds per square foot    
    para_LU21_d91[3]  = (109 + 109 + 109) / 3
    para_LU21_d91[4]  = ((0.25 + 0.5 + 2) / 3) * 0.0459137  # Tons per acre to pounds per square foot        
    para_LU21_d91[5]  = (30 + 30 + 30) / 3
    para_LU21_d91[6]  = ((0 + 1 + 0) / 3) * 0.0459137  # Tons per acre to pounds per square foot
    para_LU21_d91[7]  = (0 + 0 + 1) / 3
    para_LU21_d91[10] = ((0.5 + 4 + 0.5) / 3) * 0.0459137 # Tons per acre to pounds per square foot
    para_LU21_d91[12] = ((1.5 + 3.5 + 2) / 3) * 0.0459137 # Tons per acre to pounds per square foot #1988 addition
    para_LU21_d91[13] = (0.4 + 0.4 + 3) / 3
    para_LU21_d91[14] = (25 + 30 + 25) / 3
    para_LU21_d91[15] = (8000 + 8000 + 8000) / 3
    para_LU21_d91[16] = (25 + 14 + 59) / 3
    
    #    else:
    para_LU21_else     = np.empty((17))
    para_LU21_else[0]  = ((1.5 + 1 + 2) / 3) * 0.0459137     # Tons per acre to pounds per square foot
    para_LU21_else[1]  = (2000 + 1750 + 1500) / 3
    para_LU21_else[2]  = ((0.5 + 2.5 + 0.5) / 3) * 0.0459137 # Tons per acre to pounds per square foot
    para_LU21_else[3]  = (109 + 109 + 109) / 3
    para_LU21_else[4]  = ((0.25 + 0.5 + 2) / 3) * 0.0459137 # Tons per acre to pounds per square foot
    para_LU21_else[5]  = (30 + 30 + 30) / 3
    para_LU21_else[6]  = ((1 + 0 + 0) / 3) * 0.0459137  # Tons per acre to pounds per square foot
    para_LU21_else[7]  = (0 + 0 + 1) / 3
    para_LU21_else[10] = ((0.5 + 4 + 0.5) / 3) * 0.0459137   # Tons per acre to pounds per square foot
    para_LU21_else[12] = ((3.5 + 0.5 + 2) / 3) * 0.0459137 # Tons per acre to pounds per square foot #1988 addition
    para_LU21_else[13] = (0.4 + 0.4 + 3) / 3
    para_LU21_else[14] = (25 + 30 + 25) / 3
    para_LU21_else[15] = (8000 + 8000 + 8000) / 3
    para_LU21_else[16] = (25 + 14 + 59) / 3
    
    #    LU21:
    para_LU21     = np.empty((17))
    para_LU21[8]  = ((0.5 + 4 + 0.5) / 3) * 0.0459137 # Tons per acre to pounds per square foot
    para_LU21[9]  = (2000 + 1500 + 2000) / 3
    para_LU21[11] = (2000 + 1500 + 0) / 3
    
    ######################################################
    #elif LU_indices[index_y, index_x] == 22:  # Shrubland
    para_LU22     = np.empty((17))
    para_LU22[0]  = (2.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU22[1]  = 700
    para_LU22[2]  = (2 * 0.0459137)   # Tons per acre to pounds per square foot
    para_LU22[3]  = 109
    para_LU22[4]  = (1.5 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU22[5]  = 30
    para_LU22[6]  = (0 * 0.0459137)  # Tons per acre to pounds per square foot
    para_LU22[7]  = 0
    para_LU22[8]  = (0 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU22[9]  = 0
    para_LU22[10] = (9 * 0.0459137) # Tons per acre to pounds per square foot
    para_LU22[11] = 1250
    para_LU22[13] = 4.5
    para_LU22[14] = 15
    para_LU22[15] = 9500
    para_LU22[16] = 24

    # This loop stores the parameters for each grid point, and removes many
    # if-test with LU_indices from the grid loop.
    # In the case of LU20 and LU21 that also depend on days, these tests must be done in the grid loop

    para_yx = np.empty((17, number_of_latitudinal_degrees, number_of_longitudinal_degrees ))

    for i in prange(len_i): # prange for parallel computing
        index_x = all_indexes_lon_np[i]
        index_y = all_indexes_lat_np[i]

        if LU_indices[index_y, index_x] == 13:  # Tundra
            for k in range(17):
                para_yx[ k, index_y, index_x] = para_LU13[k]
        elif LU_indices[index_y, index_x] == 17:  # Grassland
            for k in range(17):
                para_yx[ k, index_y, index_x] = para_LU17[k]
        elif LU_indices[index_y, index_x] == 19:  # Deciduous and evergreen needleleaf forest
            for k in range(17):
                para_yx[ k, index_y, index_x] = para_LU19[k]
        elif LU_indices[index_y, index_x] == 20:  # Deciduous and evergreen broadleaf forest
            para_yx[14, index_y, index_x] = para_LU20[14]
            para_yx[15, index_y, index_x] = para_LU20[15]
            para_yx[ 8, index_y, index_x] = para_LU20[8]
            para_yx[ 9, index_y, index_x] = para_LU20[9]
            para_yx[11, index_y, index_x] = para_LU20[11]
        elif LU_indices[index_y, index_x] == 21:  # Mixed forest
            para_yx[ 8, index_y, index_x] = para_LU21[8]
            para_yx[ 9, index_y, index_x] = para_LU21[9]
            para_yx[11, index_y, index_x] = para_LU21[11]
        elif LU_indices[index_y, index_x] == 22:  # Shrubland
            for k in range(17):
                para_yx[ k, index_y, index_x] = para_LU22[k]

    return (para_yx,
            para_LU20_d91,
            para_LU20_else,
            para_LU21_d91,
            para_LU21_else)

@njit(fastmath=True, boundscheck=True)
def Slope(
    len_i, all_indexes_lon_np, all_indexes_lat_np, elevation,
    number_of_latitudinal_degrees, number_of_longitudinal_degrees):
    
    # Computing the slopes used in the grid loop.
    # It is important to do this computing only once outside the loops.
    # Define the slope array used in the grid loop.
    slope_yx = np.empty(( number_of_latitudinal_degrees, number_of_longitudinal_degrees ))

    # The loop to compute slope_yx.                                                                    
    for i in prange(len_i): # prange for parallel computing
        index_x = all_indexes_lon_np[i]
        index_y = all_indexes_lat_np[i]

        if ( #1
            index_y > 0
            and index_x > 0
            and index_y < number_of_latitudinal_degrees - 1
            and index_x < number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
                np.abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
            ]
        elif ( #2
            index_y == 0
            and index_x > 0
            and index_y < number_of_latitudinal_degrees - 1
            and index_x < number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
                np.abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
            ]
        elif ( #3
            index_y > 0
            and index_x == 0
            and index_y < number_of_latitudinal_degrees - 1
            and index_x < number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
            ]
        elif ( #4
            index_y > 0
            and index_x > 0
            and index_y == number_of_latitudinal_degrees - 1
            and index_x < number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
                np.abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
            ]
        elif ( #5
            index_y > 0
            and index_x > 0
            and index_y < number_of_latitudinal_degrees - 1
            and index_x == number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
                np.abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
            ]
        elif ( #6
            index_y == 0
            and index_x == 0
            and index_y < number_of_latitudinal_degrees - 1
            and index_x < number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
            ]
        elif ( #7
            index_y > 0
            and index_x > 0
            and index_y == number_of_latitudinal_degrees - 1
            and index_x == number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
                np.abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
            ]
        elif ( #8
            index_y == 0
            and index_x > 0
            and index_y < number_of_latitudinal_degrees - 1
            and index_x == number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x - 1]),
                np.abs(elevation[index_y, index_x] - elevation[index_y + 1, index_x]),
            ]
        elif ( #9
            index_y > 0
            and index_x == 0
            and index_y == number_of_latitudinal_degrees - 1
            and index_x < number_of_longitudinal_degrees - 1
        ):
            elevations = [
                np.abs(elevation[index_y, index_x] - elevation[index_y - 1, index_x]),
                np.abs(elevation[index_y, index_x] - elevation[index_y, index_x + 1]),
            ]

        slope_yx[index_y, index_x] = np.arctan((max(elevations)) / 3000)

    return (slope_yx)

@njit(fastmath=True, boundscheck=True)
def Cpara(
    len_i, all_indexes_lon_np, all_indexes_lat_np,
    number_of_latitudinal_degrees, number_of_longitudinal_degrees,
    precipitation_annual_input):

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

    Cpara_yx = np.empty((13,number_of_latitudinal_degrees, number_of_longitudinal_degrees))

    for i in prange(len_i): # prange for parallel computing
        index_x = all_indexes_lon_np[i]
        index_y = all_indexes_lat_np[i]
        precipitation_annual = precipitation_annual_input[index_y, index_x]

        if precipitation_annual > 2000:
            Cpara_yx[0,index_y, index_x] = 4
        elif precipitation_annual <= 2000 and precipitation_annual > 350:
            Cpara_yx[0,index_y, index_x] = 3
        elif precipitation_annual <= 350 and precipitation_annual > 200:
            Cpara_yx[0,index_y, index_x] = 2
        else:
            Cpara_yx[0,index_y, index_x] = 1
            
        if Cpara_yx[0,index_y, index_x] == 1:
            Cpara_yx[ 1,index_y, index_x] = -70
            Cpara_yx[ 2,index_y, index_x] = 12.8
            Cpara_yx[ 3,index_y, index_x] = -150.5
            Cpara_yx[ 4,index_y, index_x] = 18.4
            Cpara_yx[ 5,index_y, index_x] = 11.2
            Cpara_yx[ 6,index_y, index_x] = 7.4
            Cpara_yx[ 7,index_y, index_x] = 50  # percent
            Cpara_yx[ 8,index_y, index_x] = 12.5
            Cpara_yx[ 9,index_y, index_x] = 7.5
            Cpara_yx[10,index_y, index_x] = 0.25
            Cpara_yx[11,index_y, index_x] = 1
        elif Cpara_yx[0,index_y, index_x] == 2:
            Cpara_yx[ 1,index_y, index_x] = -100
            Cpara_yx[ 2,index_y, index_x] = 14
            Cpara_yx[ 3,index_y, index_x] = -187.7
            Cpara_yx[ 4,index_y, index_x] = 19.6
            Cpara_yx[ 5,index_y, index_x] = -10.3
            Cpara_yx[ 6,index_y, index_x] = 8.3
            Cpara_yx[ 7,index_y, index_x] = 60  # percent                       
            Cpara_yx[ 8,index_y, index_x] = -5
            Cpara_yx[ 9,index_y, index_x] = 8.2
            Cpara_yx[10,index_y, index_x] = 0.25
            Cpara_yx[11,index_y, index_x] = 2
        elif Cpara_yx[0,index_y, index_x] == 3:
            Cpara_yx[ 1,index_y, index_x] = -137.5
            Cpara_yx[ 2,index_y, index_x] = 15.5
            Cpara_yx[ 3,index_y, index_x] = -245.2
            Cpara_yx[ 4,index_y, index_x] = 22
            Cpara_yx[ 5,index_y, index_x] = -42.7
            Cpara_yx[ 6,index_y, index_x] = 9.8
            Cpara_yx[ 7,index_y, index_x] = 70  # percent
            Cpara_yx[ 8,index_y, index_x] = -22.5
            Cpara_yx[ 9,index_y, index_x] = 8.9
            Cpara_yx[10,index_y, index_x] = 0.05
            Cpara_yx[11,index_y, index_x] = 3
        elif Cpara_yx[0,index_y, index_x] == 4:
            Cpara_yx[ 1,index_y, index_x] = -185
            Cpara_yx[ 2,index_y, index_x] = 17.4
            Cpara_yx[ 3,index_y, index_x] = -305.2
            Cpara_yx[ 4,index_y, index_x] = 24.3
            Cpara_yx[ 5,index_y, index_x] = -93.5
            Cpara_yx[ 6,index_y, index_x] = 12.2
            Cpara_yx[ 7,index_y, index_x] = 80  # percent
            Cpara_yx[ 8,index_y, index_x] = -45
            Cpara_yx[ 9,index_y, index_x] = 9.8
            Cpara_yx[10,index_y, index_x] = 0.05
            Cpara_yx[11,index_y, index_x] = 4

        Cpara_yx[12,index_y, index_x] = 7 * Cpara_yx[11,index_y, index_x]

    return (Cpara_yx)

@njit(fastmath=True, boundscheck=True)
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

@njit(fastmath=True, boundscheck=True)
def MoistureContentMin(humidities_min, temperature_max):
    equilibrium_moisture_content_min = 0.0
    if humidities_min < 10:
        equilibrium_moisture_content_min = (
            0.03229
            + 0.281073 * humidities_min
            - 0.000578 * temperature_max * humidities_min
        )
    elif humidities_min >= 10 and humidities_min < 50:
        equilibrium_moisture_content_min = (
            2.22749 + 0.160107 * humidities_min - 0.014784 * temperature_max
        )
    elif humidities_min >= 50:
        equilibrium_moisture_content_min = (
            21.0606
            + 0.005565 * humidities_min**2
            - 0.00035 * humidities_min * temperature_max
            - 0.483199 * humidities_min
        )
    return equilibrium_moisture_content_min

@njit(fastmath=True, boundscheck=True)
def MoistureContentMean(humidities_mean, temperature_mean):
    equilibrium_moisture_content_mean = 0.0
    if humidities_mean < 10:
        equilibrium_moisture_content_mean = (
            0.03229
            + 0.281073 * humidities_mean
            - 0.000578 * temperature_mean * humidities_mean
        )
    elif humidities_mean >= 10 and humidities_mean < 50:
        equilibrium_moisture_content_mean = (
            2.22749 + 0.160107 * humidities_mean - 0.014784 * temperature_mean
        )
    elif humidities_mean >= 50:
        equilibrium_moisture_content_mean = (
            21.0606
            + 0.005565 * humidities_mean**2
            - 0.00035 * humidities_mean * temperature_mean
            - 0.483199 * humidities_mean
        )
    return equilibrium_moisture_content_mean

@njit(fastmath=True, boundscheck=True)
def MoistureContentMax(humidities_max, temperature_min):
    equilibrium_moisture_content_max = 0.0
    if humidities_max < 10:
        equilibrium_moisture_content_max = (
            0.03229
            + 0.281073 * humidities_max
            - 0.000578 * temperature_min * humidities_max
        )
    elif humidities_max >= 10 and humidities_max < 50:
        equilibrium_moisture_content_max = (
            2.22749 + 0.160107 * humidities_max - 0.014784 * temperature_min
        )
    elif humidities_max >= 50:
        equilibrium_moisture_content_max = (
            21.0606
            + 0.005565 * humidities_max**2
            - 0.00035 * humidities_max * temperature_min
            - 0.483199 * humidities_max
        )
    return equilibrium_moisture_content_max

@njit(fastmath=True, boundscheck=True)
def DeadOneHourFuelMoisture(equilibrium_moisture_content_mean, precipitation_duration):
    onehour_fuel_moisture = 0.0
    if precipitation_duration == 0:
        onehour_fuel_moisture = 1.03 * equilibrium_moisture_content_mean
    else:
        onehour_fuel_moisture = 35
    return onehour_fuel_moisture

@njit(fastmath=True, boundscheck=True)
def DeadTenHourFuelMoisture(equilibrium_moisture_content_mean):
    tenhour_fuel_moisture = 1.28 * equilibrium_moisture_content_mean
    return tenhour_fuel_moisture

@njit(fastmath=True, boundscheck=True)
def PrecipitationDuration(precipitation, rainfall_rate):
    precipitation_duration = np.round((precipitation / rainfall_rate) + 0.49)
    if precipitation_duration > 8:
        precipitation_duration = 8
    return precipitation_duration

@njit(fastmath=True, boundscheck=True)
def MoistureContentWeightedAverage(
    duration_daylight,
    equilibrium_moisture_content_max,
    equilibrium_moisture_content_min,
):
    moisture_content_weighted_average = (
        (duration_daylight * equilibrium_moisture_content_min)
        + ((24 - duration_daylight) * equilibrium_moisture_content_max)
    ) / 24
    return moisture_content_weighted_average

@njit(fastmath=True, boundscheck=True)
def DeadHundredHourFuelMoisture(
    previous_day_hundred_hour_fuel_moisture,
    precipitation_duration,
    moisture_content_weighted_average,
):
    boundary_condition_weighted_average_hundred = (
        ((24 - precipitation_duration) * moisture_content_weighted_average)
        + (precipitation_duration * (0.5 * precipitation_duration + 41))
    ) / 24
    dead_hundred_hour_fuel_moisture = previous_day_hundred_hour_fuel_moisture + (
        boundary_condition_weighted_average_hundred
        - previous_day_hundred_hour_fuel_moisture
    ) * (1.0 - 0.87 * math.exp(-0.24))
    return dead_hundred_hour_fuel_moisture

@njit(fastmath=True, boundscheck=True)
def BoundaryConditionWeightedAverageThousandHour(
    precipitation_duration, moisture_content_weighted_average
):
    boundary_condition_weighted_average_thousand = (
        ((24 - precipitation_duration) * moisture_content_weighted_average)
        + (precipitation_duration * (2.7 * precipitation_duration + 76.0))
    ) / 24
    return boundary_condition_weighted_average_thousand

@njit(fastmath=True, boundscheck=True)
def BoundaryConditionSevenDayAverageThousandHour(
    boundary_condition_weighted_average_thousand,
    previous_day_boundary_condition_weighted_average_thousand,
    second_previous_day_boundary_condition_weighted_average_thousand,
    third_previous_day_boundary_condition_weighted_average_thousand,
    fourth_previous_day_boundary_condition_weighted_average_thousand,
    fifth_previous_day_boundary_condition_weighted_average_thousand,
    sixth_previous_day_boundary_condition_weighted_average_thousand,
    previous_day_boundary_condition_seven_day,
    day,
    year,
    startyear,
    climate_class,
):
    boundary_condition_seven_day = 0.0
    if (day + 1) % 7 == 0:
        boundary_condition_seven_day = (
            boundary_condition_weighted_average_thousand
            + previous_day_boundary_condition_weighted_average_thousand
            + second_previous_day_boundary_condition_weighted_average_thousand
            + third_previous_day_boundary_condition_weighted_average_thousand
            + fourth_previous_day_boundary_condition_weighted_average_thousand
            + fifth_previous_day_boundary_condition_weighted_average_thousand
            + sixth_previous_day_boundary_condition_weighted_average_thousand
        ) / 7
    elif year == startyear and day == 0:
        boundary_condition_seven_day = 10.0 + (5.0 * climate_class)
    else:
        boundary_condition_seven_day = previous_day_boundary_condition_seven_day
    return boundary_condition_seven_day

@njit(fastmath=True, boundscheck=True)
def DeadThousandHourFuelMoisture(
    previous_day_thousand_hour_fuel_moisture,
    boundary_condition_seven_day,
    day,
):
    day = day + 1
    if day % 7 == 0:
        dead_thousand_hour_fuel_moisture = previous_day_thousand_hour_fuel_moisture + (
            boundary_condition_seven_day - previous_day_thousand_hour_fuel_moisture
        ) * (1.0 - 0.82 * math.exp(-0.168))
    else:
        dead_thousand_hour_fuel_moisture = previous_day_thousand_hour_fuel_moisture
    return dead_thousand_hour_fuel_moisture

@njit(fastmath=True, boundscheck=True)
def SurfaceWeightedCharacteristicAreaVolumeRatio(
    fuel_model_woody_fuel_loading,
    surface_area_volume_one_hour,
    surface_area_volume_hundred_hour,
    surface_area_volume_ten_hour,
    surface_area_volume_herbaceous,
    surface_area_volume_woody,
    fuel_particle_density,
    one_hour_fuel_loading,
    fuel_model_ten_hour_fuel_loading,
    fuel_model_hundred_hour_fuel_loading,
    herbaceous_fuel_loading,
):
    surface_area_one_hour = (
        one_hour_fuel_loading / fuel_particle_density
    ) * surface_area_volume_one_hour
    surface_area_ten_hour = (
        fuel_model_ten_hour_fuel_loading / fuel_particle_density
    ) * surface_area_volume_ten_hour
    surface_area_hundred_hour = (
        fuel_model_hundred_hour_fuel_loading / fuel_particle_density
    ) * surface_area_volume_hundred_hour
    surface_area_dead = (
        surface_area_one_hour + surface_area_ten_hour + surface_area_hundred_hour
    )

    if surface_area_dead > 0:
        weighting_factor_one_hour = surface_area_one_hour / surface_area_dead
        weighting_factor_ten_hour = surface_area_ten_hour / surface_area_dead
        weighting_factor_hundred_hour = surface_area_hundred_hour / surface_area_dead
    else:
        weighting_factor_one_hour = 0.0
        weighting_factor_ten_hour = 0.0
        weighting_factor_hundred_hour = 0.0

    surface_area_herbaceous = 0.0
    surface_area_woody = 0.0
    if fuel_particle_density > 0:
        surface_area_herbaceous = (
            herbaceous_fuel_loading / fuel_particle_density
        ) * surface_area_volume_herbaceous
        surface_area_woody = (
            fuel_model_woody_fuel_loading / fuel_particle_density
        ) * surface_area_volume_woody

    surface_area_live = surface_area_herbaceous + surface_area_woody

    weighting_factor_herbaceous = 0.0
    weighting_factor_woody = 0.0
    if surface_area_live > 0:
        weighting_factor_herbaceous = surface_area_herbaceous / surface_area_live
        weighting_factor_woody = surface_area_woody / surface_area_live

    fuel_characteristic_surface_area_volume_ratio_live = (
        weighting_factor_herbaceous * surface_area_volume_herbaceous
    ) + (weighting_factor_woody * surface_area_volume_woody)

    weighting_factor_dead = 0.0
    weighting_factor_live = 0.0
    if surface_area_dead + surface_area_live > 0:
        weighting_factor_dead = surface_area_dead / (surface_area_dead + surface_area_live)
        weighting_factor_live = surface_area_live / (surface_area_dead + surface_area_live)

    fuel_characteristic_surface_area_volume_ratio_dead = (
        (weighting_factor_one_hour * surface_area_volume_one_hour)
        + (weighting_factor_ten_hour * surface_area_volume_ten_hour)
        + (weighting_factor_hundred_hour * surface_area_volume_hundred_hour)
    )

    surface_area_weighted_characteristic_surface_area_to_volume_ratio = (
        weighting_factor_dead * fuel_characteristic_surface_area_volume_ratio_dead
    ) + (weighting_factor_live * fuel_characteristic_surface_area_volume_ratio_live)

    if not np.isfinite(surface_area_weighted_characteristic_surface_area_to_volume_ratio):
        surface_area_weighted_characteristic_surface_area_to_volume_ratio = 0.0

    return surface_area_weighted_characteristic_surface_area_to_volume_ratio

@njit(fastmath=True, boundscheck=True)
def SpreadComponent(
    net_one_hour_fuel_loading,
    net_ten_hour_fuel_loading,
    net_hundred_hour_fuel_loading,
    net_herbaceous_fuel_loading,
    net_woody_fuel_loading,
    one_hour_fuel_loading,
    fuel_model_ten_hour_fuel_loading,
    fuel_model_hundred_hour_fuel_loading,
    herbaceous_fuel_loading,
    fuel_model_woody_fuel_loading,
    fuel_particle_density,
    surface_area_volume_one_hour,
    surface_area_volume_hundred_hour,
    surface_area_volume_ten_hour,
    surface_area_volume_herbaceous,
    surface_area_volume_woody,
    packing_ratio,
    dead_one_hour_fuel_moisture,
    dead_ten_hour_fuel_moisture,
    dead_hundred_hour_fuel_moisture,
    heating_number_one_hour,
    heating_number_ten_hour,
    heating_number_hundred_hour,
    ratio_heating_numbers,
    moisture_content_herbaceous,
    moisture_content_woody,
    dead_fuel_extinction_moisture,
    slope,
    fuel_heat_combustion,
    mineral_damping_coefficient,
    windspeeds,
    bulk_density_fuel_bed,
    surface_area_weighted_characteristic_surface_area_to_volume_ratio,
):
    if (
        not math.isfinite(surface_area_weighted_characteristic_surface_area_to_volume_ratio)
        or surface_area_weighted_characteristic_surface_area_to_volume_ratio <= 0
    ):
        return 0
    else:
        surface_area_one_hour = (
            one_hour_fuel_loading / fuel_particle_density
        ) * surface_area_volume_one_hour
        surface_area_ten_hour = (
            fuel_model_ten_hour_fuel_loading / fuel_particle_density
        ) * surface_area_volume_ten_hour
        surface_area_hundred_hour = (
            fuel_model_hundred_hour_fuel_loading / fuel_particle_density
        ) * surface_area_volume_hundred_hour

        surface_area_herbaceous = 0.0
        surface_area_woody = 0.0
        if fuel_particle_density > 0:
            surface_area_herbaceous = (
                herbaceous_fuel_loading / fuel_particle_density
            ) * surface_area_volume_herbaceous
            surface_area_woody = (
                fuel_model_woody_fuel_loading / fuel_particle_density
            ) * surface_area_volume_woody

        surface_area_dead = (
            surface_area_one_hour + surface_area_ten_hour + surface_area_hundred_hour
        )
        surface_area_live = surface_area_herbaceous + surface_area_woody
        
        weighting_factor_herbaceous = 0.0
        weighting_factor_woody = 0.0
        if surface_area_live > 0:
            weighting_factor_herbaceous = surface_area_herbaceous / surface_area_live
            weighting_factor_woody = surface_area_woody / surface_area_live

        weighted_net_loading_live = (
            weighting_factor_woody * net_woody_fuel_loading
        ) + (weighting_factor_herbaceous * net_herbaceous_fuel_loading)
        weighted_moisture_content_live = (
            weighting_factor_herbaceous * moisture_content_herbaceous
        ) + (weighting_factor_woody * moisture_content_woody)


        weighting_factor_one_hour = 0.0
        weighting_factor_ten_hour = 0.0
        weighting_factor_hundred_hour = 0.0
        if surface_area_dead > 0:
            weighting_factor_one_hour = surface_area_one_hour / surface_area_dead
            weighting_factor_ten_hour = surface_area_ten_hour / surface_area_dead
            weighting_factor_hundred_hour = surface_area_hundred_hour / surface_area_dead

        weighting_factor_dead = 0.0
        weighting_factor_live = 0.0
        if surface_area_dead + surface_area_live > 0:
            weighting_factor_dead = surface_area_dead / (surface_area_dead + surface_area_live)
            weighting_factor_live = surface_area_live / (surface_area_dead + surface_area_live)

        weighted_net_loading_dead = (
            (weighting_factor_one_hour * net_one_hour_fuel_loading)
            + (weighting_factor_ten_hour * net_ten_hour_fuel_loading)
            + (weighting_factor_hundred_hour * net_hundred_hour_fuel_loading)
        )

        optimum_packing_ratio = (
            3.348
            * surface_area_weighted_characteristic_surface_area_to_volume_ratio ** (-0.8189)
        )
        maximum_reaction_velocity = (
            surface_area_weighted_characteristic_surface_area_to_volume_ratio**1.5
        ) / (
            495.0
            + 0.0594
            * surface_area_weighted_characteristic_surface_area_to_volume_ratio**1.5
        )

        weighted_optimum_reaction_velocity_exponent = (
            133.0
            * surface_area_weighted_characteristic_surface_area_to_volume_ratio ** (-0.7913)
        )
        optimum_reaction_velocity = (
            maximum_reaction_velocity
            * (packing_ratio / optimum_packing_ratio)
            ** weighted_optimum_reaction_velocity_exponent
            * math.exp(
                weighted_optimum_reaction_velocity_exponent * 1.0
                - packing_ratio / optimum_packing_ratio
            )
        )

        no_wind_propagating_flux_ratio = math.exp(
            (
                0.792
                + 0.681
                * surface_area_weighted_characteristic_surface_area_to_volume_ratio**0.5
            )
            * (packing_ratio + 0.1)
        ) / (
            192.0
            + 0.2595 * surface_area_weighted_characteristic_surface_area_to_volume_ratio
        )

        weighted_dead_moisture_content_live_extinction_moisture = (
            (dead_one_hour_fuel_moisture * heating_number_one_hour)
            + (dead_ten_hour_fuel_moisture * heating_number_ten_hour)
            + (dead_hundred_hour_fuel_moisture * heating_number_hundred_hour)
        ) / (
            heating_number_one_hour + heating_number_ten_hour + heating_number_hundred_hour
        )

        live_fuel_extinction_moisture = (
            2.9
            * ratio_heating_numbers
            * (
                1.0
                - weighted_dead_moisture_content_live_extinction_moisture
                / dead_fuel_extinction_moisture
            )
            - 0.226
        ) * 100.0
        if live_fuel_extinction_moisture < dead_fuel_extinction_moisture:
            live_fuel_extinction_moisture = dead_fuel_extinction_moisture

        weighted_moisture_content_dead = (
            weighting_factor_one_hour * dead_one_hour_fuel_moisture
            + weighting_factor_ten_hour * dead_ten_hour_fuel_moisture
            + weighting_factor_hundred_hour * dead_hundred_hour_fuel_moisture
        )

        dead_moisture_extinction_ratio = (
            weighted_moisture_content_dead / dead_fuel_extinction_moisture
        )
        live_moisture_extinction_ratio = (
            weighted_moisture_content_live / live_fuel_extinction_moisture
        )

        moisture_damping_coefficient_dead = (
            1.0
            - 2.59 * dead_moisture_extinction_ratio
            + 5.11 * dead_moisture_extinction_ratio**2.0
            - 3.52 * dead_moisture_extinction_ratio**3.0
        )
        moisture_damping_coefficient_live = (
            1.0
            - 2.59 * live_moisture_extinction_ratio
            + 5.11 * live_moisture_extinction_ratio**2.0
            - 3.52 * live_moisture_extinction_ratio**3.0
        )

        if moisture_damping_coefficient_dead < 0:
            moisture_damping_coefficient_dead = 0
        elif moisture_damping_coefficient_dead > 1:
            moisture_damping_coefficient_dead = 1

        if moisture_damping_coefficient_live < 0:
            moisture_damping_coefficient_live = 0
        elif moisture_damping_coefficient_live > 1:
            moisture_damping_coefficient_live = 1

        reaction_intensity = optimum_reaction_velocity * (
            (
                weighted_net_loading_dead
                * fuel_heat_combustion
                * mineral_damping_coefficient
                * moisture_damping_coefficient_dead
            )
            + (
                weighted_net_loading_live
                * fuel_heat_combustion
                * mineral_damping_coefficient
                * moisture_damping_coefficient_live
            )
        )

        if reaction_intensity < 0:
            reaction_intensity = 0

        wind_effect_exponent_b = (
            0.02526
            * surface_area_weighted_characteristic_surface_area_to_volume_ratio**0.54
        )
        wind_effect_coefficient_c = 7.47 * math.exp(
            -0.133 * surface_area_weighted_characteristic_surface_area_to_volume_ratio**0.55
        )
        wind_effect_exponent_e = 0.715 * math.exp(
            -3.59
            * 10.0 ** (-4.0)
            * surface_area_weighted_characteristic_surface_area_to_volume_ratio
        )
        wind_effect_coefficient_u = wind_effect_coefficient_c * (
            packing_ratio / optimum_packing_ratio
        ) ** (-wind_effect_exponent_e)

        wind_effect_multiplier = 0.0
        if (windspeeds * 88.0) <= (0.9 * reaction_intensity):
            wind_effect_multiplier = (
                wind_effect_coefficient_u * (windspeeds * 88.0) ** wind_effect_exponent_b
            )
        else:
            wind_effect_multiplier = (
                wind_effect_coefficient_u
                * (0.9 * reaction_intensity) ** wind_effect_exponent_b
            )

        slope_effect_multiplier_coefficient = 5.275 * (math.tan(slope)) ** 2.0
        slope_effect_multiplier = slope_effect_multiplier_coefficient * packing_ratio ** (
            -0.3
        )

        if (
            surface_area_volume_one_hour > 0
            and surface_area_volume_ten_hour > 0
            and surface_area_volume_hundred_hour > 0
        ):
            dead_term = weighting_factor_dead * (
                weighting_factor_one_hour
                * math.exp(-138.0 / surface_area_volume_one_hour)
                * (250.0 + 11.16 * dead_one_hour_fuel_moisture)
                + weighting_factor_ten_hour
                * math.exp(-138.0 / surface_area_volume_ten_hour)
                * (250.0 + 11.16 * dead_ten_hour_fuel_moisture)
                + weighting_factor_hundred_hour
                * math.exp(-138.0 / surface_area_volume_hundred_hour)
                * (250.0 + 11.16 * dead_hundred_hour_fuel_moisture)
            )
        elif (
            surface_area_volume_one_hour == 0
            and surface_area_volume_ten_hour > 0
            and surface_area_volume_hundred_hour > 0
        ):
            dead_term = weighting_factor_dead * (
                weighting_factor_ten_hour
                * math.exp(-138.0 / surface_area_volume_ten_hour)
                * (250.0 + 11.16 * dead_ten_hour_fuel_moisture)
                + weighting_factor_hundred_hour
                * math.exp(-138.0 / surface_area_volume_hundred_hour)
                * (250.0 + 11.16 * dead_hundred_hour_fuel_moisture)
            )
        elif (
            surface_area_volume_one_hour > 0
            and surface_area_volume_ten_hour == 0
            and surface_area_volume_hundred_hour > 0
        ):
            dead_term = weighting_factor_dead * (
                weighting_factor_one_hour
                * math.exp(-138.0 / surface_area_volume_one_hour)
                * (250.0 + 11.16 * dead_one_hour_fuel_moisture)
                + weighting_factor_hundred_hour
                * math.exp(-138.0 / surface_area_volume_hundred_hour)
                * (250.0 + 11.16 * dead_hundred_hour_fuel_moisture)
            )
        elif (
            surface_area_volume_one_hour > 0
            and surface_area_volume_ten_hour > 0
            and surface_area_volume_hundred_hour == 0
        ):
            dead_term = weighting_factor_dead * (
                weighting_factor_one_hour
                * math.exp(-138.0 / surface_area_volume_one_hour)
                * (250.0 + 11.16 * dead_one_hour_fuel_moisture)
                + weighting_factor_ten_hour
                * math.exp(-138.0 / surface_area_volume_ten_hour)
                * (250.0 + 11.16 * dead_ten_hour_fuel_moisture)
            )
        elif (
            surface_area_volume_one_hour == 0
            and surface_area_volume_ten_hour == 0
            and surface_area_volume_hundred_hour > 0
        ):
            dead_term = weighting_factor_dead * (
                weighting_factor_hundred_hour
                * math.exp(-138.0 / surface_area_volume_hundred_hour)
                * (250.0 + 11.16 * dead_hundred_hour_fuel_moisture)
            )
        elif (
            surface_area_volume_one_hour == 0
            and surface_area_volume_ten_hour > 0
            and surface_area_volume_hundred_hour == 0
        ):
            dead_term = weighting_factor_dead * (
                weighting_factor_ten_hour
                * math.exp(-138.0 / surface_area_volume_ten_hour)
                * (250.0 + 11.16 * dead_ten_hour_fuel_moisture)
            )
        elif (
            surface_area_volume_one_hour > 0
            and surface_area_volume_ten_hour == 0
            and surface_area_volume_hundred_hour == 0
        ):
            dead_term = weighting_factor_dead * (
                weighting_factor_one_hour
                * math.exp(-138.0 / surface_area_volume_one_hour)
                * (250.0 + 11.16 * dead_one_hour_fuel_moisture)
            )
        else:
            dead_term = 0

        if (
            surface_area_volume_woody > 0
            and surface_area_volume_herbaceous > 0
        ):
            live_term = weighting_factor_live * (
                weighting_factor_herbaceous
                * math.exp(-138.0 / surface_area_volume_herbaceous)
                * (250.0 + 11.16 * moisture_content_herbaceous)
                + weighting_factor_woody
                * math.exp(-138.0 / surface_area_volume_woody)
                * (250.0 + 11.16 * moisture_content_woody)
            )
        elif (
            surface_area_volume_woody == 0
            and surface_area_volume_herbaceous > 0
        ):
            live_term = weighting_factor_live * (
                weighting_factor_herbaceous
                * math.exp(-138.0 / surface_area_volume_herbaceous)
                * (250.0 + 11.16 * moisture_content_herbaceous)
            )
        elif (
            surface_area_volume_woody > 0
            and surface_area_volume_herbaceous == 0
        ):
            live_term = weighting_factor_live * (
                weighting_factor_woody
                * math.exp(-138.0 / surface_area_volume_woody)
                * (250.0 + 11.16 * moisture_content_woody)
            )
        else:
            live_term = 0

        # --- Final heat sink ---
        heat_sink = bulk_density_fuel_bed * (dead_term + live_term)
        
        if heat_sink <= 0:
            spread_component = 0.0
        else:
            heat_sink = max(heat_sink, 1e-4)
            
            spread_component = 0.0
            if heat_sink > 0:
                spread_component = np.round(
                    reaction_intensity
                    * no_wind_propagating_flux_ratio
                    * (1.0 + slope_effect_multiplier + wind_effect_multiplier)
                    / heat_sink
                )  # ft/min
        return spread_component
    

@njit(fastmath=True, boundscheck=True)
def EnergyReleaseComponent(
    one_hour_fuel_loading,
    fuel_model_ten_hour_fuel_loading,
    fuel_model_hundred_hour_fuel_loading,
    fuel_model_thousand_hour_fuel_loading,
    total_dead_fuel_loading,
    total_live_fuel_loading,
    total_fuel_loading,
    net_woody_fuel_loading,
    net_herbaceous_fuel_loading,
    packing_ratio,
    surface_area_volume_one_hour,
    surface_area_volume_ten_hour,
    surface_area_volume_hundred_hour,
    surface_area_volume_thousand_hour,
    surface_area_volume_woody,
    surface_area_volume_herbaceous,
    dead_fuel_extinction_moisture,
    heating_number_one_hour,
    heating_number_ten_hour,
    heating_number_hundred_hour,
    ratio_heating_numbers,
    fuel_heat_combustion,
    mineral_damping_coefficient,
    moisture_content_herbaceous,
    moisture_content_woody,
    dead_one_hour_fuel_moisture,
    dead_ten_hour_fuel_moisture,
    dead_hundred_hour_fuel_moisture,
    dead_thousand_hour_fuel_moisture,
    surface_area_weighted_characteristic_surface_area_to_volume_ratio,
):
    weighting_factor_one_hour = 0.0
    weighting_factor_ten_hour = 0.0
    weighting_factor_hundred_hour = 0.0
    weighting_factor_thousand_hour = 0.0
    
    if total_dead_fuel_loading > 0:
        weighting_factor_one_hour = one_hour_fuel_loading / total_dead_fuel_loading
        weighting_factor_ten_hour = (
            fuel_model_ten_hour_fuel_loading / total_dead_fuel_loading
        )
        weighting_factor_hundred_hour = (
            fuel_model_hundred_hour_fuel_loading / total_dead_fuel_loading
        )
        weighting_factor_thousand_hour = (
            fuel_model_thousand_hour_fuel_loading / total_dead_fuel_loading
        )

    weighting_factor_herbaceous = 0.0
    weighting_factor_woody = 0.0
    if total_live_fuel_loading > 0:
        weighting_factor_herbaceous = (
            net_herbaceous_fuel_loading / total_live_fuel_loading
        )
        weighting_factor_woody = net_woody_fuel_loading / total_live_fuel_loading
    
    characteristic_surface_area_to_volume_ratio_live = (
        weighting_factor_woody * surface_area_volume_woody
    ) + (weighting_factor_herbaceous * surface_area_volume_herbaceous)
    
    weighted_moisture_content_live = (
        weighting_factor_herbaceous * moisture_content_herbaceous
    ) + (weighting_factor_woody * moisture_content_woody)

    weighting_factor_dead = 0.0
    weighting_factor_live = 0.0
    if total_fuel_loading > 0:
        weighting_factor_dead = total_dead_fuel_loading / total_fuel_loading
        weighting_factor_live = total_live_fuel_loading / total_fuel_loading

    characteristic_surface_area_to_volume_ratio_dead = (
        (weighting_factor_one_hour * surface_area_volume_one_hour)
        + (weighting_factor_ten_hour * surface_area_volume_ten_hour)
        + (weighting_factor_hundred_hour * surface_area_volume_hundred_hour)
        + (weighting_factor_thousand_hour * surface_area_volume_thousand_hour)
    )

    mass_weighted_characteristic_surface_area_to_volume_ratio = (
        weighting_factor_dead * characteristic_surface_area_to_volume_ratio_dead
    ) + (weighting_factor_live * characteristic_surface_area_to_volume_ratio_live)

    optimum_packing_ratio = (
        3.348 * mass_weighted_characteristic_surface_area_to_volume_ratio ** (-0.8189)
    )
    maximum_reaction_velocity = (
        mass_weighted_characteristic_surface_area_to_volume_ratio** 1.5
        / (
            495.0
            + 0.0594 * mass_weighted_characteristic_surface_area_to_volume_ratio**1.5
        )
    )

    weighted_optimum_reaction_velocity_exponent = (
        133.0 * mass_weighted_characteristic_surface_area_to_volume_ratio ** (-0.7913)
    )
    optimum_reaction_velocity = (
        maximum_reaction_velocity
        * (packing_ratio / optimum_packing_ratio)
        ** weighted_optimum_reaction_velocity_exponent
        * math.exp(
            weighted_optimum_reaction_velocity_exponent
            * (1.0 - packing_ratio / optimum_packing_ratio)
        )
    )

    weighted_moisture_content_dead = (
        (weighting_factor_one_hour * dead_one_hour_fuel_moisture)
        + (weighting_factor_ten_hour * dead_ten_hour_fuel_moisture)
        + (weighting_factor_hundred_hour * dead_hundred_hour_fuel_moisture)
        + (weighting_factor_thousand_hour * dead_thousand_hour_fuel_moisture)
    )

    weighted_dead_moisture_content_live_extinction_moisture = (
        (dead_one_hour_fuel_moisture * heating_number_one_hour)
        + (dead_ten_hour_fuel_moisture * heating_number_ten_hour)
        + (dead_hundred_hour_fuel_moisture * heating_number_hundred_hour)
    ) / (
        heating_number_one_hour + heating_number_ten_hour + heating_number_hundred_hour
    )

    live_fuel_extinction_moisture = (
        2.9
        * ratio_heating_numbers
        * (
            1.0
            - weighted_dead_moisture_content_live_extinction_moisture
            / dead_fuel_extinction_moisture
        )
        - 0.226
    ) * 100.0
    if live_fuel_extinction_moisture < dead_fuel_extinction_moisture:
        live_fuel_extinction_moisture = dead_fuel_extinction_moisture

    dead_moisture_extinction_ratio = 0.0
    if dead_fuel_extinction_moisture > 0:
        dead_moisture_extinction_ratio = (
            weighted_moisture_content_dead / dead_fuel_extinction_moisture
        )
    live_moisture_extinction_ratio = 0.0
    if live_fuel_extinction_moisture > 0:
        live_moisture_extinction_ratio = (
            weighted_moisture_content_live / live_fuel_extinction_moisture
        )

    moisture_damping_coefficient_dead = (
        1.0
        - 2.0 * dead_moisture_extinction_ratio
        + 1.5 * dead_moisture_extinction_ratio**2.0
        - 0.5 * dead_moisture_extinction_ratio**3.0
    )
    if moisture_damping_coefficient_dead < 0:
        moisture_damping_coefficient_dead = 0
    elif moisture_damping_coefficient_dead > 1:
        moisture_damping_coefficient_dead = 1

    moisture_damping_coefficient_live = (
        1.0
        - 2.0 * live_moisture_extinction_ratio
        + 1.5 * live_moisture_extinction_ratio**2.0
        - 0.5 * live_moisture_extinction_ratio**3.0
    )
    if moisture_damping_coefficient_live < 0:
        moisture_damping_coefficient_live = 0
    elif moisture_damping_coefficient_live > 1:
        moisture_damping_coefficient_live = 1

    net_dead_fuel_loading = total_dead_fuel_loading * (1 - 0.0555)
    net_live_fuel_loading = total_live_fuel_loading * (1 - 0.0555)

    reaction_intensity = optimum_reaction_velocity * (
        weighting_factor_dead
        * net_dead_fuel_loading
        * fuel_heat_combustion
        * mineral_damping_coefficient
        * moisture_damping_coefficient_dead
    ) + (
        weighting_factor_live
        * net_live_fuel_loading
        * fuel_heat_combustion
        * mineral_damping_coefficient
        * moisture_damping_coefficient_live
    )
    if surface_area_weighted_characteristic_surface_area_to_volume_ratio > 0:
        residence_time_flaming_front = (
            384.0 / surface_area_weighted_characteristic_surface_area_to_volume_ratio
        )
    else:
        residence_time_flaming_front = 0.0

    energy_release_component = 0.04 * reaction_intensity * residence_time_flaming_front
    energy_release_component = np.round(energy_release_component)
    return energy_release_component

@njit(fastmath=True, boundscheck=True)
def BurningIndex(spread_component, energy_release_component):
    burning_index = np.round(3.01 * (spread_component * energy_release_component) ** 0.46)
    return burning_index
