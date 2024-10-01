from typing import Dict, List, Tuple

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma
from scipy.stats import halfnorm

from . import data as dat

def running_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def flour_aniso(g_factor, intensity_para, intensity_perp, l1_japan_corr, l2_japan_corr):
    """Fluorescence Anisotropy calculation.

    See equation 7 in Kudryavtsev, V., Sikor, M., Kalinin, S., Mokranjac, D., Seidel, C.A.M. and Lamb, D.C. (2012),
    Combining MFD and PIE for Accurate Single-Pair Förster Resonance Energy Transfer Measurements. ChemPhysChem, 13: 1060-1078.
    https://doi.org/10.1002/cphc.201100822
    """
    numerator = g_factor * intensity_para - intensity_perp
    denominator = ((1 - 3 * l2_japan_corr) * g_factor * intensity_para +
                   (2 - 3 * l1_japan_corr) * intensity_perp)
    return numerator / denominator

def interphoton_arrival_times(all_macro_times, all_micro_times, macro_res, micro_res):
    # - Each detected photon has a time of detection encoded by the macro time + the micro time. **all_macro_times** and **all_micro_times** are arrays whose index is represents the detected photons in order of detection, while the value represents the associated macro or micro time for each photon.
    # - **macro_res** and **micro_res** represent the resolution of the macro and micro times in seconds.
    # - The **macro time** indicates the time in units of **macro_res** that the excitation laser was last fired directly before this photon was detected.
    # - The **micro time** indicates the amount of time in units of **micro_res** that has elapsed since the excitation laser was last fired at which the photon was detected, i.e. it's the amount of time elapsed from the macro time at which the photon was detected.
    # - The interphoton arrival time is calculated by iterating through **all_macro_times** and **all_micro_times** and calculating the time elapsed between each photon detection event.
    
    arr_size = len(all_macro_times) - 1
    photon_time_intervals = np.zeros(arr_size, dtype=np.float64)
    for i in range(arr_size):
        photon_1 = (all_macro_times[i]*macro_res) + (all_micro_times[i]*micro_res)
        photon_2 = (all_macro_times[i+1]*macro_res) + (all_micro_times[i+1]*micro_res)
        photon_time_intervals[i] = (photon_2 - photon_1)*1000  # Convert to ms
    photon_ids = np.arange(1, arr_size + 1)
    return photon_time_intervals, photon_ids

def estimate_background_noise(logrunavg, bins_y):
    counts_logrunavg, bins_logrunavg, _ = plt.hist(logrunavg, bins=bins_y, alpha=0.6, color='r')
    plt.close()  # Close the plot to prevent it from displaying during function call
    index_max = np.argmax(counts_logrunavg)
    noise_mean = bins_logrunavg[index_max]*0.95
    filtered_logrunavg = ma.masked_less(logrunavg, noise_mean).compressed()
    mu, std = halfnorm.fit(filtered_logrunavg)
    return mu, std, noise_mean, filtered_logrunavg, bins_logrunavg
