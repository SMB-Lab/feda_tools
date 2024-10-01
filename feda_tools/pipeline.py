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
import numpy.ma as ma

from feda_tools.core import utilities as utils
from feda_tools.core import analysis as an
from feda_tools.core import data as dat
from feda_tools.core import model
from feda_tools.core import process

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
    data_ptu, data_irf, data_bkg = dat.load_ptu_files(file_ptu, file_irf, file_bkg)

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
    photon_time_intervals, photon_ids = an.interphoton_arrival_times(
        all_macro_times, all_micro_times, macro_res, micro_res
    )
    print(f"Last photon time interval: {photon_time_intervals[-1]} ms")

    # Calculate running average
    window_size = 30
    running_avg = an.running_average(photon_time_intervals, window_size)
    xarr = np.arange(window_size - 1, len(photon_time_intervals))
    logrunavg = np.log10(running_avg)

    # Estimate background noise
    bins = {"x": 141, "y": 141}
    bins_y = bins['y']
    mu, std, noise_mean, filtered_logrunavg, bins_logrunavg = an.estimate_background_noise(logrunavg, bins_y)

    # Define threshold value
    threshold_value = mu - 4 * std

    # Extract bursts
    burst_index, filtered_values = dat.extract_greater(logrunavg, threshold_value)

    # Setup fit23
    counts_irf, _ = np.histogram(all_micro_times_irf, bins=num_bins)
    counts_irf_nb = counts_irf.copy()
    counts_irf_nb[0:3] = 0
    counts_irf_nb[10:66] = 0
    counts_irf_nb[74:128] = 0
    fit23 = model.setup_fit23(num_bins, macro_res, counts_irf_nb, g_factor, l1_japan_corr, l2_japan_corr)

    # Process bursts
    bi4_bur_df, bg4_df = process.process_bursts(
        burst_index, all_macro_times, all_micro_times, routing_channels, macro_res, micro_res,
        min_photon_count, bg4_micro_time_min, bg4_micro_time_max, g_factor, l1_japan_corr,
        l2_japan_corr, bg4_bkg_para, bg4_bkg_perp, fit23
    )

    # Save results
    output_directory = r'./tests/burstid_selection_viz_tool'
    dat.save_results(output_directory, file_path, bi4_bur_df, bg4_df)

if __name__ == '__main__':
    main()