from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as p
import pathlib
import tttrlib
import os
from feda_tools.core import twodim_hist as tdh
from feda_tools.core import utilities as utils
from feda_tools.core import analysis as an
from decimal import Decimal, getcontext
import numpy.ma as ma
from scipy.stats import norm
from scipy.stats import halfnorm

def calc_flour_aniso(g_factor, intensity_para, intensity_perp, l1_japan_corr, l2_japan_corr):
    """Fluorescence Anisotropy calculation.

    See equation 7 in Kudryavtsev, V., Sikor, M., Kalinin, S., Mokranjac, D., Seidel, C.A.M. and Lamb, D.C. (2012),
    Combining MFD and PIE for Accurate Single-Pair Förster Resonance Energy Transfer Measurements. ChemPhysChem, 13: 1060-1078.
    https://doi.org/10.1002/cphc.201100822
    """
    numerator = g_factor * intensity_para - intensity_perp
    denominator = ((1 - 3 * l2_japan_corr) * g_factor * intensity_para +
                   (2 - 3 * l1_japan_corr) * intensity_perp)
    return numerator / denominator

def load_ptu_files(file_ptu, file_irf, file_bkg):
    data_ptu = tttrlib.TTTR(file_ptu, 'PTU')
    data_irf = tttrlib.TTTR(file_irf, 'PTU')
    data_bkg = tttrlib.TTTR(file_bkg, 'PTU')
    return data_ptu, data_irf, data_bkg

def calculate_interphoton_arrival_times(all_macro_times, all_micro_times, macro_res, micro_res):
    arr_size = len(all_macro_times) - 1
    photon_time_intervals = np.zeros(arr_size, dtype=np.float64)
    for i in range(arr_size):
        photon_1 = (all_macro_times[i]*macro_res) + (all_micro_times[i]*micro_res)
        photon_2 = (all_macro_times[i+1]*macro_res) + (all_micro_times[i+1]*micro_res)
        photon_time_intervals[i] = (photon_2 - photon_1)*1000  # Convert to ms
    photon_ids = np.arange(1, arr_size + 1)
    return photon_time_intervals, photon_ids

def running_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def estimate_background_noise(logrunavg, bins_y):
    counts_logrunavg, bins_logrunavg, _ = plt.hist(logrunavg, bins=bins_y, alpha=0.6, color='r')
    plt.close()  # Close the plot to prevent it from displaying during function call
    index_max = np.argmax(counts_logrunavg)
    noise_mean = bins_logrunavg[index_max]*0.95
    filtered_logrunavg = ma.masked_less(logrunavg, noise_mean).compressed()
    mu, std = halfnorm.fit(filtered_logrunavg)
    return mu, std, noise_mean, filtered_logrunavg, bins_logrunavg

def extract_unmasked_indices(masked_array):
    unmasked_indices_lists = []
    current_indices = []
    for i, value in enumerate(masked_array):
        if np.ma.is_masked(value):
            if current_indices:
                unmasked_indices_lists.append(current_indices)
                current_indices = []
        else:
            current_indices.append(i)
    if current_indices:
        unmasked_indices_lists.append(current_indices)
    return unmasked_indices_lists

def extract_bursts(logrunavg, threshold_value):
    filtered_values = ma.masked_greater(logrunavg, threshold_value)
    burst_index = extract_unmasked_indices(filtered_values)
    return burst_index, filtered_values

def setup_fit23(num_bins, macro_res, counts_irf_nb, g_factor, l1_japan_corr, l2_japan_corr):
    dt = 25000/num_bins/1000
    period = 1/(macro_res*np.power(10, 6))
    fit23 = tttrlib.Fit23(
        dt=dt,
        irf=counts_irf_nb,
        background=np.ones_like(counts_irf_nb)*0.002,
        period=period,
        g_factor=g_factor,
        l1=l1_japan_corr,
        l2=l2_japan_corr,
        convolution_stop=10,
        p2s_twoIstar_flag=True
    )
    return fit23

def process_bursts(burst_index, all_macro_times, all_micro_times, routing_channels, macro_res, micro_res,
                   min_photon_count, bg4_micro_time_min, bg4_micro_time_max, g_factor, l1_japan_corr,
                   l2_japan_corr, bg4_bkg_para, bg4_bkg_perp, fit23):
    bi4_bur_df = pd.DataFrame()
    bg4_df = pd.DataFrame()
    bg4_channel_2_photons_total = []
    bg4_channel_0_photons_total = []
    for burst in tqdm(burst_index):
        if len(burst) <= min_photon_count:
            continue
        first_photon = burst[0]
        last_photon = burst[-1]
        lp_time = all_macro_times[last_photon]*macro_res + all_micro_times[last_photon]*micro_res
        fp_time = all_macro_times[first_photon]*macro_res + all_micro_times[first_photon]*micro_res
        lp_time_ms = lp_time*1000
        fp_time_ms = fp_time*1000
        duration = lp_time_ms - fp_time_ms
        macro_times = all_macro_times[burst]*macro_res*1000
        mean_macro_time = np.mean(macro_times)
        num_photons = len(burst)
        count_rate = num_photons / duration if duration != 0 else np.nan
        list_of_indexes = burst
        mask_channel_0 = routing_channels[list_of_indexes] == 0
        mask_channel_2 = routing_channels[list_of_indexes] == 2
        indexes_channel_0 = np.array(list_of_indexes)[mask_channel_0]
        indexes_channel_2 = np.array(list_of_indexes)[mask_channel_2]
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
        if first_green_photon is not None and last_green_photon is not None:
            lgp_time = all_macro_times[last_green_photon]*macro_res + all_micro_times[last_green_photon]*micro_res
            fgp_time = all_macro_times[first_green_photon]*macro_res + all_micro_times[first_green_photon]*micro_res
            lgp_time_ms = lgp_time*1000
            fgp_time_ms = fgp_time*1000
            duration_green = lgp_time_ms - fgp_time_ms
        else:
            duration_green = np.nan
        macro_times_ch0 = all_macro_times[indexes_channel_0]*macro_res*1000
        macro_times_ch2 = all_macro_times[indexes_channel_2]*macro_res*1000
        combined_macro_times = np.concatenate([macro_times_ch0, macro_times_ch2], axis=0)
        mean_macro_time_green = np.mean(combined_macro_times) if len(combined_macro_times) > 0 else np.nan
        num_photons_gr = len(indexes_channel_0) + len(indexes_channel_2)
        count_rate_gr = num_photons_gr / duration_green if duration_green != 0 else np.nan
        bur_new_row = {
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
        }
        bur_new_record = pd.DataFrame.from_dict(bur_new_row)
        bi4_bur_df = pd.concat([bi4_bur_df, bur_new_record], ignore_index=True)
        bg4_channel_2_photons = [index for index in burst if routing_channels[index] == 2 and
                                 bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max]
        bg4_channel_2_count = len(bg4_channel_2_photons)
        bg4_channel_2_photons_total.extend(bg4_channel_2_photons)
        bg4_channel_0_photons = [index for index in burst if routing_channels[index] == 0 and
                                 bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max]
        bg4_channel_0_count = len(bg4_channel_0_photons)
        bg4_channel_0_photons_total.extend(bg4_channel_0_photons)
        bg4_total_count = bg4_channel_2_count + bg4_channel_0_count
        bg4_rexp = calc_flour_aniso(g_factor, bg4_channel_2_count, bg4_channel_0_count,
                                    l1_japan_corr, l2_japan_corr)
        bg4_rscat = calc_flour_aniso(
            g_factor,
            bg4_channel_2_count - bg4_bkg_para,
            bg4_channel_0_count - bg4_bkg_perp,
            l1_japan_corr,
            l2_japan_corr
        )
        counts = np.array([bg4_channel_0_count, bg4_channel_2_count])
        tau, gamma, r0, rho = 3.03, 0.02, 0.38, 1.64
        x0 = np.array([tau, gamma, r0, rho])
        fixed = np.array([0, 0, 1, 0])
        r2 = fit23(data=counts, initial_values=x0, fixed=fixed, include_model=True)
        fit_tau = r2['x'][0]
        fit_gamma = r2['x'][1]
        fit_r0 = r2['x'][2]
        fit_rho = r2['x'][3]
        fit_softbifl = r2['x'][4]
        fit_2istar = r2['x'][5]
        fit_rs_scatter = r2['x'][6]
        fit_rs_exp = r2['x'][7]
        bg4_new_row = {
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
        }
        bg4_new_record = pd.DataFrame.from_dict(bg4_new_row)
        bg4_df = pd.concat([bg4_df, bg4_new_record], ignore_index=True)
    return bi4_bur_df, bg4_df

def save_results(output_directory, file_path, bi4_bur_df, bg4_df):
    bur_filename = os.path.splitext(os.path.basename(str(file_path)))[0]
    bur_filepath = os.path.join(output_directory, bur_filename) + ".bur"
    bi4_bur_df.to_csv(bur_filepath, sep='\t', index=False, float_format='%.6f')
    bg4_filepath = os.path.join(output_directory, bur_filename) + ".bg4"
    bg4_df.to_csv(bg4_filepath, sep='\t', index=False, float_format='%.6f')

def main():
    # Set file paths
    file_path = pathlib.Path('./test data/2024')
    file_source = '/Split_20230809_HighFRETDNAStd_1hr_Dani-000000.ptu'
    file_water = '/20230809_IRFddH2O_Dani_5min.ptu'
    file_buffer = '/20230809_bg_HighFRETDNAStd_30sec_Dani.ptu'
    file_ptu = str(file_path) + file_source
    file_irf = str(file_path) + file_water
    file_bkg = str(file_path) + file_buffer

    if not file_path.exists():
        raise FileNotFoundError("The provided testing path does not exist")

    # Load PTU Files
    data_ptu, data_irf, data_bkg = load_ptu_files(file_ptu, file_irf, file_bkg)

    # Extract data
    all_macro_times = data_ptu.macro_times
    all_micro_times = data_ptu.micro_times
    routing_channels = data_ptu.routing_channels

    all_macro_times_irf = data_irf.macro_times
    all_micro_times_irf = data_irf.micro_times
    routing_channels_irf = data_irf.routing_channels

    all_macro_times_bkg = data_bkg.macro_times
    all_micro_times_bkg = data_bkg.micro_times
    routing_channels_bkg = data_bkg.routing_channels

    # Get resolutions
    macro_res = data_ptu.get_header().macro_time_resolution
    micro_res = data_ptu.get_header().micro_time_resolution

    # Total duration in seconds
    total_duration = all_macro_times[-1] * macro_res

    # Define analysis window for subset of PTU
    min_event = 0
    max_event = 300000

    # Analysis settings
    min_photon_count = 60
    bg4_micro_time_min = 0
    bg4_micro_time_max = 12499
    g_factor = 1.04
    l1_japan_corr = 0.0308
    l2_japan_corr = 0.0368
    bg4_bkg_para = 0
    bg4_bkg_perp = 0
    num_bins = 128
    bin_width = macro_res / micro_res / num_bins / 1000

    # Calculate interphoton arrival times
    photon_time_intervals, photon_ids = calculate_interphoton_arrival_times(
        all_macro_times, all_micro_times, macro_res, micro_res
    )
    print(f"Last photon time interval: {photon_time_intervals[-1]} ms")

    # Calculate running average
    window_size = 30
    running_avg = running_average(photon_time_intervals, window_size)
    xarr = np.arange(window_size - 1, len(photon_time_intervals))
    logrunavg = np.log10(running_avg)

    # Estimate background noise
    bins = {"x": 141, "y": 141}
    bins_y = bins['y']
    mu, std, noise_mean, filtered_logrunavg, bins_logrunavg = estimate_background_noise(logrunavg, bins_y)

    # Define threshold value
    threshold_value = mu - 4 * std

    # Extract bursts
    burst_index, filtered_values = extract_bursts(logrunavg, threshold_value)

    # Setup fit23
    counts_irf, _ = np.histogram(all_micro_times_irf, bins=num_bins)
    counts_irf_nb = counts_irf.copy()
    counts_irf_nb[0:3] = 0
    counts_irf_nb[10:66] = 0
    counts_irf_nb[74:128] = 0
    fit23 = setup_fit23(num_bins, macro_res, counts_irf_nb, g_factor, l1_japan_corr, l2_japan_corr)

    # Process bursts
    bi4_bur_df, bg4_df = process_bursts(
        burst_index, all_macro_times, all_micro_times, routing_channels, macro_res, micro_res,
        min_photon_count, bg4_micro_time_min, bg4_micro_time_max, g_factor, l1_japan_corr,
        l2_japan_corr, bg4_bkg_para, bg4_bkg_perp, fit23
    )

    # Save results
    output_directory = r'./tests/burstid_selection_viz_tool'
    save_results(output_directory, file_path, bi4_bur_df, bg4_df)

if __name__ == '__main__':
    main()