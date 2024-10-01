import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from feda_tools.core import analysis as an

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
        bg4_rexp = an.flour_aniso(g_factor, bg4_channel_2_count, bg4_channel_0_count,
                                    l1_japan_corr, l2_japan_corr)
        bg4_rscat = an.flour_aniso(
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