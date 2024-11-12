import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from feda_tools.core import analysis as an

def calculate_burst_timing(burst, all_macro_times, all_micro_times, macro_res, micro_res):
    """Calculate basic timing information for a burst"""
    first_photon = burst[0]
    last_photon = burst[-1]
    
    lp_time = all_macro_times[last_photon]*macro_res + all_micro_times[last_photon]*micro_res
    fp_time = all_macro_times[first_photon]*macro_res + all_micro_times[first_photon]*micro_res
    
    lp_time_ms = lp_time*1000
    fp_time_ms = fp_time*1000
    duration = lp_time_ms - fp_time_ms
    
    macro_times = all_macro_times[burst]*macro_res*1000
    mean_macro_time = np.mean(macro_times)
    
    return first_photon, last_photon, duration, mean_macro_time

def get_green_photon_info(burst, routing_channels):
    """Extract information about green channel photons"""
    mask_channel_0 = routing_channels[burst] == 0
    mask_channel_2 = routing_channels[burst] == 2
    indexes_channel_0 = np.array(burst)[mask_channel_0]
    indexes_channel_2 = np.array(burst)[mask_channel_2]
    
    if len(indexes_channel_0) > 0 and len(indexes_channel_2) > 0:
        first_green_photon = min(np.min(indexes_channel_0), np.min(indexes_channel_2))
        last_green_photon = max(np.max(indexes_channel_0), np.max(indexes_channel_2))
    elif len(indexes_channel_0) >= 2:
        first_green_photon = np.min(indexes_channel_0)
        last_green_photon = np.max(indexes_channel_0)
    elif len(indexes_channel_2) >= 2:
        first_green_photon = np.min(indexes_channel_2)
        last_green_photon = np.max(indexes_channel_2)
    else:
        first_green_photon = None
        last_green_photon = None
        
    return first_green_photon, last_green_photon, indexes_channel_0, indexes_channel_2

def calculate_green_timing(first_green_photon, last_green_photon, all_macro_times, all_micro_times, 
                         macro_res, micro_res, indexes_channel_0, indexes_channel_2):
    """Calculate timing information for green channel photons"""
    if first_green_photon is not None and last_green_photon is not None:
        lgp_time = all_macro_times[last_green_photon]*macro_res + all_micro_times[last_green_photon]*micro_res
        fgp_time = all_macro_times[first_green_photon]*macro_res + all_micro_times[first_green_photon]*micro_res
        duration_green = (lgp_time - fgp_time) * 1000
    else:
        duration_green = np.nan

    macro_times_ch0 = all_macro_times[indexes_channel_0]*macro_res*1000
    macro_times_ch2 = all_macro_times[indexes_channel_2]*macro_res*1000
    combined_macro_times = np.concatenate([macro_times_ch0, macro_times_ch2], axis=0)
    
    mean_macro_time_green = np.mean(combined_macro_times) if len(combined_macro_times) > 0 else np.nan
    return duration_green, mean_macro_time_green

def calculate_bg4_counts(burst, routing_channels, all_micro_times, bg4_micro_time_min, bg4_micro_time_max):
    """Calculate BG4 window photon counts"""
    bg4_channel_2_photons = [
        index for index in burst if routing_channels[index] == 2 and
        bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max
    ]
    bg4_channel_2_count = len(bg4_channel_2_photons)

    bg4_channel_0_photons = [
        index for index in burst if routing_channels[index] == 0 and
        bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max
    ]
    bg4_channel_0_count = len(bg4_channel_0_photons)
    
    return bg4_channel_0_count, bg4_channel_2_count

def perform_fit23_analysis(counts, initial_fit_params, fit23, first_photon, last_photon):
    """Perform fit23 analysis on the burst data"""
    x0 = initial_fit_params
    fixed = np.array([0, 0, 1, 0])

    try:
        r2 = fit23(data=counts, initial_values=x0, fixed=fixed, include_model=True)
        fit_tau, fit_gamma, fit_r0, fit_rho = r2['x'][:4]
        fit_softbifl = r2['x'][4] if len(r2['x']) > 4 else np.nan
        fit_2istar = r2['x'][5] if len(r2['x']) > 5 else np.nan
        fit_rs_scatter = r2['x'][6] if len(r2['x']) > 6 else np.nan
        fit_rs_exp = r2['x'][7] if len(r2['x']) > 7 else np.nan
        
        return fit_tau, fit_gamma, fit_r0, fit_rho, fit_softbifl, fit_2istar, fit_rs_scatter, fit_rs_exp
    except Exception as e:
        print(f"Fit23 error for burst {first_photon}-{last_photon}: {e}")
        return None

def process_single_burst(
    burst, all_macro_times, all_micro_times, routing_channels, macro_res, micro_res,
    min_photon_count, bg4_micro_time_min, bg4_micro_time_max, g_factor, l1_japan_corr,
    l2_japan_corr, bg4_bkg_para, bg4_bkg_perp, fit23, initial_fit_params
):
    """Main function to process a single burst of photons"""
    if len(burst) <= min_photon_count:
        return None, None

    # Get basic burst timing information
    first_photon, last_photon, duration, mean_macro_time = calculate_burst_timing(
        burst, all_macro_times, all_micro_times, macro_res, micro_res
    )

    # Get green photon information
    first_green_photon, last_green_photon, indexes_channel_0, indexes_channel_2 = get_green_photon_info(
        burst, routing_channels
    )

    # Calculate green timing information
    duration_green, mean_macro_time_green = calculate_green_timing(
        first_green_photon, last_green_photon, all_macro_times, all_micro_times,
        macro_res, micro_res, indexes_channel_0, indexes_channel_2
    )

    # Calculate basic statistics
    num_photons = len(burst)
    count_rate = num_photons / duration if duration != 0 else np.nan
    num_photons_gr = len(indexes_channel_0) + len(indexes_channel_2)
    count_rate_gr = num_photons_gr / duration_green if duration_green != 0 else np.nan

    # Create first DataFrame
    bi4_bur_df = pd.DataFrame({
        'First Photon': [first_photon],
        'Last Photon': [last_photon],
        'Duration (ms)': [duration],
        'Mean Macro Time (ms)': [mean_macro_time],
        'Number of Photons': [num_photons],
        'Count Rate (kHz)': [count_rate],
        'Duration (green) (ms)': [duration_green],
        'Mean Macro Time (green) (ms)': [mean_macro_time_green],
        'Number of Photons (green)': [num_photons_gr],
        'Green Count Rate (kHz)': [count_rate_gr]
    })

    # Calculate BG4 counts
    bg4_channel_0_count, bg4_channel_2_count = calculate_bg4_counts(
        burst, routing_channels, all_micro_times, bg4_micro_time_min, bg4_micro_time_max
    )

    # Calculate anisotropy
    bg4_total_count = bg4_channel_2_count + bg4_channel_0_count
    bg4_rexp = an.flour_aniso(
        g_factor, bg4_channel_2_count, bg4_channel_0_count,
        l1_japan_corr, l2_japan_corr
    )
    bg4_rscat = an.flour_aniso(
        g_factor,
        bg4_channel_2_count - bg4_bkg_para,
        bg4_channel_0_count - bg4_bkg_perp,
        l1_japan_corr,
        l2_japan_corr
    )

    # Perform fit23 analysis
    counts = np.array([bg4_channel_0_count, bg4_channel_2_count])
    fit_results = perform_fit23_analysis(counts, initial_fit_params, fit23, first_photon, last_photon)
    
    if fit_results is None:
        return None, None
        
    fit_tau, fit_gamma, fit_r0, fit_rho, fit_softbifl, fit_2istar, fit_rs_scatter, fit_rs_exp = fit_results

    # Create second DataFrame
    bg4_df = pd.DataFrame({
        'Ng-p-all': [bg4_channel_2_count],
        'Ng-s-all': [bg4_channel_0_count],
        'Number of Photons (fit window) (green)': [bg4_total_count],
        'r Scatter (green)': [bg4_rscat],
        'r Experimental (green)': [bg4_rexp],
        'Fit tau': [fit_tau],
        'Fit gamma': [fit_gamma],
        'Fit r0': [fit_r0],
        'Fit rho': [fit_rho],
        'Fit softbifl': [fit_softbifl],
        'Fit 2I*': [fit_2istar],
        'Fit rs_scatter': [fit_rs_scatter],
        'Fit rs_exp': [fit_rs_exp]
    })

    return bi4_bur_df, bg4_df
