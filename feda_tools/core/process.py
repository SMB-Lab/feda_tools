import numpy as np
import pandas as pd

from feda_tools.core import analysis as an

# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def _calc_photon_timing(index, all_macro_times, all_micro_times, macro_res, micro_res):
    """Return absolute arrival time (in seconds) for a single photon index."""
    return all_macro_times[index] * macro_res + all_micro_times[index] * micro_res


def calculate_burst_timing(burst, all_macro_times, all_micro_times, macro_res, micro_res):
    """Calculate basic timing information for a burst (all photons)."""
    first_photon = burst[0]
    last_photon = burst[-1]

    fp_time = _calc_photon_timing(first_photon, all_macro_times, all_micro_times, macro_res, micro_res)
    lp_time = _calc_photon_timing(last_photon,  all_macro_times, all_micro_times, macro_res, micro_res)

    fp_time_ms = fp_time * 1_000
    lp_time_ms = lp_time * 1_000
    duration   = lp_time_ms - fp_time_ms

    macro_times = all_macro_times[burst] * macro_res * 1_000
    mean_macro_time = np.mean(macro_times)

    return first_photon, last_photon, duration, mean_macro_time


# --------------------------------------------------------------------------------------
# Channel‑specific helpers (green = 0/2, red = 1/3)
# --------------------------------------------------------------------------------------

def _get_channel_indexes(burst, routing_channels, channels):
    mask = np.isin(routing_channels[burst], channels)
    return np.array(burst)[mask]


def _split_parallel_perp(indexes, routing_channels, ch_parallel, ch_perp):
    mask_parallel = routing_channels[indexes] == ch_parallel
    idx_parallel  = indexes[mask_parallel]
    idx_perp      = indexes[~mask_parallel]
    return idx_parallel, idx_perp


def _channel_photon_info(burst, routing_channels, chan_parallel, chan_perp):
    """Return first/last photon and indexes for the (parallel, perpendicular) detector pair."""
    idx_parallel, idx_perp = _split_parallel_perp(
        _get_channel_indexes(burst, routing_channels, (chan_parallel, chan_perp)),
        routing_channels,
        chan_parallel,
        chan_perp,
    )

    if len(idx_parallel) and len(idx_perp):
        first_idx = min(np.min(idx_parallel), np.min(idx_perp))
        last_idx  = max(np.max(idx_parallel), np.max(idx_perp))
    elif len(idx_parallel) >= 2:
        first_idx, last_idx = np.min(idx_parallel), np.max(idx_parallel)
    elif len(idx_perp) >= 2:
        first_idx, last_idx = np.min(idx_perp), np.max(idx_perp)
    else:
        first_idx, last_idx = None, None

    return first_idx, last_idx, idx_parallel, idx_perp


def _calculate_channel_timing(first_idx, last_idx, idx_parallel, idx_perp,
                              all_macro_times, all_micro_times, macro_res, micro_res):
    """Timing and mean macro‑time for a single colour channel."""
    if first_idx is not None and last_idx is not None:
        t_first = _calc_photon_timing(first_idx, all_macro_times, all_micro_times, macro_res, micro_res)
        t_last  = _calc_photon_timing(last_idx,  all_macro_times, all_micro_times, macro_res, micro_res)
        duration_ms = (t_last - t_first) * 1_000
    else:
        duration_ms = np.nan

    macro_times_parallel = all_macro_times[idx_parallel] * macro_res * 1_000
    macro_times_perp     = all_macro_times[idx_perp]     * macro_res * 1_000
    combined_macro_times = np.concatenate([macro_times_parallel, macro_times_perp]) if len(idx_parallel)+len(idx_perp) else np.array([])

    mean_macro_time_ms = np.mean(combined_macro_times) if combined_macro_times.size else np.nan
    return duration_ms, mean_macro_time_ms


# --------------------------------------------------------------------------------------
# BG4 helper (works for any pair of detector channels)
# --------------------------------------------------------------------------------------

def calculate_bg4_counts(burst, routing_channels, all_micro_times,
                          chan_parallel, chan_perp,
                          bg4_micro_time_min, bg4_micro_time_max):
    """Return (N_perp, N_para) counts inside BG4 micro‑time window for a detector pair."""
    mask_window = (bg4_micro_time_min < all_micro_times) & (all_micro_times < bg4_micro_time_max)

    idx_para = [i for i in burst if routing_channels[i] == chan_parallel   and mask_window[i]]
    idx_perp = [i for i in burst if routing_channels[i] == chan_perp      and mask_window[i]]

    return len(idx_perp), len(idx_para)   # (s‑pol, p‑pol) order kept from original code


# --------------------------------------------------------------------------------------
# Fit23 wrapper
# --------------------------------------------------------------------------------------

def perform_fit23_analysis(counts, initial_fit_params, fit23, first_photon, last_photon):
    """Same as before, with safe NaNs on failure."""
    x0     = initial_fit_params
    fixed  = np.array([0, 0, 1, 0])

    try:
        r2 = fit23(data=counts, initial_values=x0, fixed=fixed, include_model=True)
        fit_tau, fit_gamma, fit_r0, fit_rho = r2['x'][:4]
        fit_softbifl   = r2['x'][4] if len(r2['x']) > 4 else np.nan
        fit_2istar     = r2['x'][5] if len(r2['x']) > 5 else np.nan
        fit_rs_scatter = r2['x'][6] if len(r2['x']) > 6 else np.nan
        fit_rs_exp     = r2['x'][7] if len(r2['x']) > 7 else np.nan
        return fit_tau, fit_gamma, fit_r0, fit_rho, fit_softbifl, fit_2istar, fit_rs_scatter, fit_rs_exp
    except Exception as e:
        print(f"Fit23 error for burst {first_photon}-{last_photon}: {e}")
        return (np.nan,) * 8


# --------------------------------------------------------------------------------------
# Main processing routine
# --------------------------------------------------------------------------------------

def process_single_burst(
    burst,
    all_macro_times,
    all_micro_times,
    routing_channels,
    macro_res,
    micro_res,
    min_photon_count,
    bg4_micro_time_min,
    bg4_micro_time_max,
    g_factor,
    l1_japan_corr,
    l2_japan_corr,
    bg4_bkg_para,
    bg4_bkg_perp,
    fit23,
    initial_fit_params,
):
    """Process ONE burst; now returns two DataFrames with both green and red results."""

    # ---------------------------------- sanity check ----------------------------------
    if len(burst) <= min_photon_count:
        return None, None

    # -------------------------------- burst‑level stats -------------------------------
    first_photon, last_photon, duration, mean_macro_time = calculate_burst_timing(
        burst, all_macro_times, all_micro_times, macro_res, micro_res
    )

    # -------------------------------- green channel -----------------------------------
    fg, lg, idx_ch0, idx_ch2 = _channel_photon_info(burst, routing_channels, 0, 2)
    dur_g, mean_mt_g = _calculate_channel_timing(
        fg, lg, idx_ch0, idx_ch2, all_macro_times, all_micro_times, macro_res, micro_res
    )

    # -------------------------------- red channel -------------------------------------
    fr, lr, idx_ch1, idx_ch3 = _channel_photon_info(burst, routing_channels, 1, 3)
    dur_r, mean_mt_r = _calculate_channel_timing(
        fr, lr, idx_ch1, idx_ch3, all_macro_times, all_micro_times, macro_res, micro_res
    )

    # -------------------------------- photon counts & rates ---------------------------
    n_tot = len(burst)
    cr_tot = n_tot / duration if duration else np.nan

    n_g   = len(idx_ch0) + len(idx_ch2)
    cr_g  = n_g / dur_g if dur_g else np.nan

    n_r   = len(idx_ch1) + len(idx_ch3)
    cr_r  = n_r / dur_r if dur_r else np.nan

    # ------------------------------------ DF #1 ---------------------------------------
    burst_df = pd.DataFrame(
        {
            "First Photon":                [first_photon],
            "Last Photon":                 [last_photon],
            "Duration (ms)":               [duration],
            "Mean Macro Time (ms)":        [mean_macro_time],
            "Number of Photons":           [n_tot],
            "Count Rate (kHz)":            [cr_tot],
            "Duration (green) (ms)":       [dur_g],
            "Mean Macro Time (green) (ms)":[mean_mt_g],
            "Number of Photons (green)":   [n_g],
            "Green Count Rate (kHz)":      [cr_g],
            "Duration (red) (ms)":         [dur_r],
            "Mean Macro Time (red) (ms)":  [mean_mt_r],
            "Number of Photons (red)":     [n_r],
            "Red Count Rate (kHz)":        [cr_r],
        }
    )

    # -------------------------------- BG4 window: green -------------------------------
    n_g_s, n_g_p = calculate_bg4_counts(
        burst, routing_channels, all_micro_times, 0, 2, bg4_micro_time_min, bg4_micro_time_max
    )
    n_g_bg4 = n_g_s + n_g_p
    r_exp_g = an.flour_aniso(g_factor, n_g_p, n_g_s, l1_japan_corr, l2_japan_corr)
    r_scat_g = an.flour_aniso(
        g_factor,
        n_g_p - bg4_bkg_para,
        n_g_s - bg4_bkg_perp,
        l1_japan_corr,
        l2_japan_corr,
    )

    fit_g = perform_fit23_analysis(
        np.array([n_g_s, n_g_p]), initial_fit_params, fit23, first_photon, last_photon
    )
    tau_g, gamma_g, r0_g, rho_g, softbifl_g, i2star_g, rs_scat_g, rs_exp_g = fit_g

    # -------------------------------- BG4 window: red ---------------------------------
    n_r_s, n_r_p = calculate_bg4_counts(
        burst, routing_channels, all_micro_times, 1, 3, bg4_micro_time_min, bg4_micro_time_max
    )
    n_r_bg4 = n_r_s + n_r_p
    r_exp_r = an.flour_aniso(g_factor, n_r_p, n_r_s, l1_japan_corr, l2_japan_corr)
    r_scat_r = an.flour_aniso(
        g_factor,
        n_r_p - bg4_bkg_para,
        n_r_s - bg4_bkg_perp,
        l1_japan_corr,
        l2_japan_corr,
    )

    fit_r = perform_fit23_analysis(
        np.array([n_r_s, n_r_p]), initial_fit_params, fit23, first_photon, last_photon
    )
    tau_r, gamma_r, r0_r, rho_r, softbifl_r, i2star_r, rs_scat_r, rs_exp_r = fit_r

    # ------------------------------------ DF #2 ---------------------------------------
    bg4_df = pd.DataFrame(
        {
            # green
            "Ng-p-all":                          [n_g_p],
            "Ng-s-all":                          [n_g_s],
            "Number of Photons (fit window) (g)":[n_g_bg4],
            "r Scatter (green)":                 [r_scat_g],
            "r Experimental (green)":            [r_exp_g],
            "Fit tau (g)":                       [tau_g],
            "Fit gamma (g)":                     [gamma_g],
            "Fit r0 (g)":                        [r0_g],
            "Fit rho (g)":                       [rho_g],
            "Fit softbifl (g)":                  [softbifl_g],
            "Fit 2I* (g)":                       [i2star_g],
            "Fit rs_scatter (g)":                [rs_scat_g],
            "Fit rs_exp (g)":                    [rs_exp_g],
            # red
            "Nr-p-all":                          [n_r_p],
            "Nr-s-all":                          [n_r_s],
            "Number of Photons (fit window) (r)":[n_r_bg4],
            "r Scatter (red)":                   [r_scat_r],
            "r Experimental (red)":              [r_exp_r],
            "Fit tau (r)":                       [tau_r],
            "Fit gamma (r)":                     [gamma_r],
            "Fit r0 (r)":                        [r0_r],
            "Fit rho (r)":                       [rho_r],
            "Fit softbifl (r)":                  [softbifl_r],
            "Fit 2I* (r)":                       [i2star_r],
            "Fit rs_scatter (r)":                [rs_scat_r],
            "Fit rs_exp (r)":                    [rs_exp_r],
        }
    )

    return burst_df, bg4_df
