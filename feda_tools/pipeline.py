from tqdm import tqdm
import numpy as np
import pandas as pd
import pathlib
import tttrlib

from feda_tools.core import analysis as an
from feda_tools.core import data as dat
from feda_tools.core import burst_processing as bp

def main():
    # Set file paths (adapt to current directory structure)
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

    # Get resolutions
    macro_res = data_ptu.get_header().macro_time_resolution
    micro_res = data_ptu.get_header().micro_time_resolution

    print("Micro Time Resolution:", micro_res*10**12, "ps")
    print("Macro Time Resolution:", macro_res*10**9, "ns")

    # Total number of events (photons)
    total_events = len(all_macro_times)
    print("Total Events:", total_events)
    
    # Analysis window for subset of PTU (for testing)
    min_event = 0
    max_event = 300000

    # Analysis settings (matching working pipeline)
    min_photon_count = 60
    min_photon_count_green = 20
    
    # BG4 parameters (Prompt)
    bg4_micro_time_min = 1000
    bg4_micro_time_max = 7000
    
    # BR4 parameters (Prompt) 
    br4_micro_time_min = bg4_micro_time_min
    br4_micro_time_max = bg4_micro_time_max
    
    # BY4 parameters (Delay)
    by4_micro_time_min = 13500
    by4_micro_time_max = 18000
    
    # Fluorescence anisotropy parameters
    g_factor = 1.04
    g_factor_red = 2.5
    l1_japan_corr = 0.0308
    l2_japan_corr = 0.0368
    
    # Background signals
    bg4_bkg_para = 0
    bg4_bkg_perp = 0
    br4_bkg_para = 0
    br4_bkg_perp = 0
    by4_bkg_para = 0
    by4_bkg_perp = 0

    # Calculate interphoton arrival times for the full dataset
    print("Calculating interphoton arrival times...")
    photon_time_intervals, photon_ids = an.interphoton_arrival_times(
        all_macro_times, all_micro_times, macro_res, micro_res
    )

    # Calculate running average
    print("Calculating running average...")
    window_size = 30
    running_avg = an.running_average(photon_time_intervals, window_size)
    xarr = np.arange(window_size - 1, len(photon_time_intervals))
    logrunavg = np.log10(running_avg)

    # Estimate background noise (matching working pipeline)
    print("Estimating background noise...")
    bins = {"x": 141, "y": 141}
    bins_y = bins['y']
    mu, std, noise_mean, filtered_logrunavg, bins_logrunavg = an.estimate_background_noise(logrunavg, bins_y)

    # Define 4-sigma threshold for burst isolation
    threshold_value = mu - 4 * std
    print(f"Threshold value: {threshold_value}")

    # Extract bursts using threshold
    print("Extracting burst indices...")
    burst_index, filtered_values = dat.extract_greater(logrunavg, threshold_value)
    print(f"Found {len(burst_index)} potential bursts")

    # Process bursts using new burst processing module
    print("Processing bursts...")
    bi4_bur_df, bg4_df, br4_df, by4_df = bp.process_bursts(
        burst_index=burst_index,
        all_macro_times=all_macro_times,
        all_micro_times=all_micro_times,
        routing_channels=routing_channels,
        macro_res=macro_res,
        micro_res=micro_res,
        min_photon_count=min_photon_count,
        min_photon_count_green=min_photon_count_green,
        bg4_micro_time_min=bg4_micro_time_min,
        bg4_micro_time_max=bg4_micro_time_max,
        br4_micro_time_min=br4_micro_time_min,
        br4_micro_time_max=br4_micro_time_max,
        by4_micro_time_min=by4_micro_time_min,
        by4_micro_time_max=by4_micro_time_max,
        g_factor=g_factor,
        g_factor_red=g_factor_red,
        l1_japan_corr=l1_japan_corr,
        l2_japan_corr=l2_japan_corr,
        bg4_bkg_para=bg4_bkg_para,
        bg4_bkg_perp=bg4_bkg_perp,
        br4_bkg_para=br4_bkg_para,
        br4_bkg_perp=br4_bkg_perp,
        by4_bkg_para=by4_bkg_para,
        by4_bkg_perp=by4_bkg_perp
    )

    print(f"Processed {len(bi4_bur_df)} valid bursts")
    print("Columns in bi4_bur_df:", bi4_bur_df.columns.tolist())
    print("Columns in bg4_df:", bg4_df.columns.tolist()) 
    print("Columns in br4_df:", br4_df.columns.tolist())
    print("Columns in by4_df:", by4_df.columns.tolist())

    # Save results (matching working pipeline output directory structure)
    output_directory = r'./tests/burstid_selection_viz_tool'
    dat.save_results(output_directory, file_path, bi4_bur_df, bg4_df, br4_df, by4_df)

if __name__ == '__main__':
    main()
