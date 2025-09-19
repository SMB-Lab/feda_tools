"""
Burst processing module for FEDA tools.

This module contains the main burst processing logic extracted from the working pipeline,
focusing on burst identification, channel separation, and steady-state anisotropy calculations
for BG4, BR4, and BY4 output files.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from . import analysis as an


def process_bursts(
    burst_index: List[List[int]],
    all_macro_times: np.ndarray,
    all_micro_times: np.ndarray,
    routing_channels: np.ndarray,
    macro_res: float,
    micro_res: float,
    min_photon_count: int,
    min_photon_count_green: int,
    bg4_micro_time_min: int,
    bg4_micro_time_max: int,
    br4_micro_time_min: int,
    br4_micro_time_max: int,
    by4_micro_time_min: int,
    by4_micro_time_max: int,
    g_factor: float,
    g_factor_red: float,
    l1_japan_corr: float,
    l2_japan_corr: float,
    bg4_bkg_para: float = 0,
    bg4_bkg_perp: float = 0,
    br4_bkg_para: float = 0,
    br4_bkg_perp: float = 0,
    by4_bkg_para: float = 0,
    by4_bkg_perp: float = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process burst indices to generate burst statistics and anisotropy data.
    
    Args:
        burst_index: List of lists, each containing photon indices for a burst
        all_macro_times: Array of macro times for all photons
        all_micro_times: Array of micro times for all photons
        routing_channels: Array of routing channels for all photons
        macro_res: Macro time resolution in seconds
        micro_res: Micro time resolution in seconds
        min_photon_count: Minimum total photons required for burst processing
        min_photon_count_green: Minimum green photons required for processing
        bg4_micro_time_min/max: Micro time window for green (BG4) analysis
        br4_micro_time_min/max: Micro time window for red (BR4) analysis  
        by4_micro_time_min/max: Micro time window for yellow (BY4) analysis
        g_factor: G-factor for green channel anisotropy
        g_factor_red: G-factor for red channel anisotropy
        l1_japan_corr: L1 Japan correction factor
        l2_japan_corr: L2 Japan correction factor
        bg4_bkg_para/perp: Background counts for green channel
        br4_bkg_para/perp: Background counts for red channel  
        by4_bkg_para/perp: Background counts for yellow channel
        
    Returns:
        Tuple of (bi4_bur_df, bg4_df, br4_df, by4_df) DataFrames
    """
    
    # Initialize result DataFrames
    bi4_bur_df = pd.DataFrame()
    bg4_df = pd.DataFrame()
    br4_df = pd.DataFrame()
    by4_df = pd.DataFrame()
    
    # Process each burst
    for burst in burst_index:
        
        # Filter out bursts with insufficient photons
        if len(burst) <= min_photon_count:
            continue
            
        # Calculate basic burst statistics
        burst_stats = calculate_burst_statistics(
            burst, all_macro_times, all_micro_times, routing_channels,
            macro_res, micro_res
        )
        
        # Skip if not enough green photons
        if burst_stats['num_photons_green'] <= min_photon_count_green:
            continue
            
        # Calculate channel-specific anisotropy data
        bg4_data = calculate_bg4_anisotropy(
            burst, routing_channels, all_micro_times,
            bg4_micro_time_min, bg4_micro_time_max,
            g_factor, l1_japan_corr, l2_japan_corr,
            bg4_bkg_para, bg4_bkg_perp, micro_res
        )
        
        br4_data = calculate_br4_anisotropy(
            burst, routing_channels, all_micro_times,
            br4_micro_time_min, br4_micro_time_max,
            g_factor_red, l1_japan_corr, l2_japan_corr,
            br4_bkg_para, br4_bkg_perp, micro_res
        )
        
        by4_data = calculate_by4_anisotropy(
            burst, routing_channels, all_micro_times,
            by4_micro_time_min, by4_micro_time_max,
            g_factor_red, l1_japan_corr, l2_japan_corr,
            by4_bkg_para, by4_bkg_perp, micro_res
        )
        
        # Calculate mean micro time for green channel
        tau_green = calculate_mean_micro_time(
            burst, routing_channels, all_micro_times,
            bg4_micro_time_min, bg4_micro_time_max,
            micro_res
        )
        
        # Create records for each DataFrame
        bur_record = create_bur_record(burst_stats, bg4_data, br4_data, by4_data)
        bg4_record = create_bg4_record(burst, bg4_data, tau_green)
        br4_record = create_br4_record(burst, br4_data)
        by4_record = create_by4_record(burst, by4_data)
        
        # Append records to DataFrames
        bi4_bur_df = pd.concat([bi4_bur_df, bur_record], ignore_index=True)
        bg4_df = pd.concat([bg4_df, bg4_record], ignore_index=True)
        br4_df = pd.concat([br4_df, br4_record], ignore_index=True)
        by4_df = pd.concat([by4_df, by4_record], ignore_index=True)
    
    return bi4_bur_df, bg4_df, br4_df, by4_df


def calculate_burst_statistics(
    burst: List[int],
    all_macro_times: np.ndarray,
    all_micro_times: np.ndarray,
    routing_channels: np.ndarray,
    macro_res: float,
    micro_res: float
) -> Dict[str, Any]:
    """Calculate basic burst statistics (timing, counts, rates)."""
    
    # First and last photon indices
    first_photon = burst[0]
    last_photon = burst[-1]
    
    # Calculate duration in ms
    lp_time = all_macro_times[last_photon] * macro_res + all_micro_times[last_photon] * micro_res
    fp_time = all_macro_times[first_photon] * macro_res + all_micro_times[first_photon] * micro_res
    duration = (lp_time - fp_time) * 1000  # Convert to ms
    
    # Mean macro time in ms  
    macro_times = all_macro_times[first_photon] * macro_res * 1000
    mean_macro_time = np.mean(macro_times)
    
    # Total photon count
    num_photons = len(burst)
    
    # Count rate in kHz
    count_rate = num_photons / duration if duration > 0 else 0
    
    # Channel-specific analysis
    channel_stats = calculate_channel_statistics(
        burst, routing_channels, all_macro_times, all_micro_times,
        macro_res, micro_res
    )
    
    return {
        'first_photon': first_photon,
        'last_photon': last_photon,
        'duration': duration,
        'mean_macro_time': mean_macro_time,
        'num_photons': num_photons,
        'count_rate': count_rate,
        **channel_stats
    }


def calculate_channel_statistics(
    burst: List[int],
    routing_channels: np.ndarray,
    all_macro_times: np.ndarray,
    all_micro_times: np.ndarray,
    macro_res: float,
    micro_res: float
) -> Dict[str, Any]:
    """Calculate channel-specific statistics for green and red channels."""
    
    # Get channel indices
    mask_channel_0 = routing_channels[burst] == 0  # Green parallel
    mask_channel_2 = routing_channels[burst] == 2  # Green perpendicular
    mask_channel_1 = routing_channels[burst] == 1  # Red parallel
    mask_channel_3 = routing_channels[burst] == 3  # Red perpendicular
    
    # Channel indices
    indexes_channel_0 = np.array(burst)[mask_channel_0]
    indexes_channel_2 = np.array(burst)[mask_channel_2]
    indexes_channel_1 = np.array(burst)[mask_channel_1]
    indexes_channel_3 = np.array(burst)[mask_channel_3]
    
    # Green channel statistics
    green_stats = calculate_color_channel_stats(
        indexes_channel_0, indexes_channel_2,
        all_macro_times, all_micro_times, macro_res, micro_res
    )
    
    # Red channel statistics  
    red_stats = calculate_color_channel_stats(
        indexes_channel_1, indexes_channel_3,
        all_macro_times, all_micro_times, macro_res, micro_res
    )
    
    return {
        'duration_green': green_stats['duration'],
        'mean_macro_time_green': green_stats['mean_macro_time'],
        'num_photons_green': green_stats['num_photons'],
        'count_rate_green': green_stats['count_rate'],
        'duration_red': red_stats['duration'],
        'mean_macro_time_red': red_stats['mean_macro_time'],
        'num_photons_red': red_stats['num_photons'],
        'count_rate_red': red_stats['count_rate'],
    }


def calculate_color_channel_stats(
    indexes_parallel: np.ndarray,
    indexes_perpendicular: np.ndarray,
    all_macro_times: np.ndarray,
    all_micro_times: np.ndarray,
    macro_res: float,
    micro_res: float
) -> Dict[str, Any]:
    """Calculate statistics for a specific color channel (parallel + perpendicular)."""
    
    # Find first and last photons for this color
    first_photon, last_photon = None, None
    
    if len(indexes_parallel) > 0 and len(indexes_perpendicular) > 0:
        first_photon = min(np.min(indexes_parallel), np.min(indexes_perpendicular))
        last_photon = max(np.max(indexes_parallel), np.max(indexes_perpendicular))
    elif len(indexes_parallel) >= 2:
        first_photon = np.min(indexes_parallel)
        last_photon = np.max(indexes_parallel)
    elif len(indexes_perpendicular) >= 2:
        first_photon = np.min(indexes_perpendicular)
        last_photon = np.max(indexes_perpendicular)
    
    # Calculate duration
    if first_photon is not None and last_photon is not None:
        lgp_time = all_macro_times[last_photon] * macro_res + all_micro_times[last_photon] * micro_res
        fgp_time = all_macro_times[first_photon] * macro_res + all_micro_times[first_photon] * micro_res
        duration = (lgp_time - fgp_time) * 1000  # Convert to ms
    else:
        duration = np.nan
    
    # Calculate mean macro time
    macro_times_parallel = all_macro_times[indexes_parallel] * macro_res * 1000 if len(indexes_parallel) > 0 else np.array([])
    macro_times_perpendicular = all_macro_times[indexes_perpendicular] * macro_res * 1000 if len(indexes_perpendicular) > 0 else np.array([])
    combined_macro_times = np.concatenate([macro_times_parallel, macro_times_perpendicular])
    
    mean_macro_time = np.mean(combined_macro_times) if len(combined_macro_times) > 0 else np.nan
    
    # Count photons and calculate rate
    num_photons = len(indexes_parallel) + len(indexes_perpendicular)
    count_rate = num_photons / duration if not np.isnan(duration) and duration > 0 else np.nan
    
    return {
        'duration': duration,
        'mean_macro_time': mean_macro_time,
        'num_photons': num_photons,
        'count_rate': count_rate
    }


def calculate_bg4_anisotropy(
    burst: List[int],
    routing_channels: np.ndarray,
    all_micro_times: np.ndarray,
    bg4_micro_time_min: int,
    bg4_micro_time_max: int,
    g_factor: float,
    l1_japan_corr: float,
    l2_japan_corr: float,
    bg4_bkg_para: float,
    bg4_bkg_perp: float,
    micro_res: float
) -> Dict[str, Any]:
    """Calculate BG4 (green channel) anisotropy data."""
    
    # Find photons in green channels within BG4 micro time window
    bg4_channel_2_photons = [
        index for index in burst 
        if routing_channels[index] == 2 and 
        bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max
    ]
    
    bg4_channel_0_photons = [
        index for index in burst
        if routing_channels[index] == 0 and
        bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max
    ]
    
    bg4_channel_2_count = len(bg4_channel_2_photons)
    bg4_channel_0_count = len(bg4_channel_0_photons)
    bg4_total_count = bg4_channel_2_count + bg4_channel_0_count
    
    # Calculate duration and signal (using micro_res for proper time conversion)
    bg4_duration = (bg4_micro_time_max - bg4_micro_time_min) * micro_res  # Convert to seconds
    bg4_signal_green = (bg4_total_count / bg4_duration) / 1000  # Convert to kHz
    
    # Calculate anisotropy
    r_exp = an.flour_aniso(g_factor, bg4_channel_2_count, bg4_channel_0_count, l1_japan_corr, l2_japan_corr)
    r_scat = an.flour_aniso(
        g_factor, 
        bg4_channel_2_count - bg4_bkg_para, 
        bg4_channel_0_count - bg4_bkg_perp, 
        l1_japan_corr, l2_japan_corr
    )
    
    return {
        'ng_p_all': bg4_channel_2_count,
        'ng_s_all': bg4_channel_0_count,
        'total_count': bg4_total_count,
        'signal_khz': bg4_signal_green,
        'r_exp': r_exp,
        'r_scat': r_scat,
        'photon_indices_para': bg4_channel_2_photons,
        'photon_indices_perp': bg4_channel_0_photons
    }


def calculate_br4_anisotropy(
    burst: List[int],
    routing_channels: np.ndarray,
    all_micro_times: np.ndarray,
    br4_micro_time_min: int,
    br4_micro_time_max: int,
    g_factor_red: float,
    l1_japan_corr: float,
    l2_japan_corr: float,
    br4_bkg_para: float,
    br4_bkg_perp: float,
    micro_res: float
) -> Dict[str, Any]:
    """Calculate BR4 (red channel) anisotropy data."""
    
    # Find photons in red channels within BR4 micro time window
    br4_channel_3_photons = [
        index for index in burst
        if routing_channels[index] == 3 and
        br4_micro_time_min < all_micro_times[index] < br4_micro_time_max
    ]
    
    br4_channel_1_photons = [
        index for index in burst
        if routing_channels[index] == 1 and
        br4_micro_time_min < all_micro_times[index] < br4_micro_time_max
    ]
    
    br4_channel_3_count = len(br4_channel_3_photons)
    br4_channel_1_count = len(br4_channel_1_photons)
    br4_total_count = br4_channel_3_count + br4_channel_1_count
    
    # Calculate duration and signal (using micro_res for proper time conversion)
    br4_duration = (br4_micro_time_max - br4_micro_time_min) * micro_res  # Convert to seconds
    br4_signal_red = (br4_total_count / br4_duration) / 1000  # Convert to kHz
    
    # Calculate anisotropy
    r_exp = an.flour_aniso(g_factor_red, br4_channel_3_count, br4_channel_1_count, l1_japan_corr, l2_japan_corr)
    r_scat = an.flour_aniso(
        g_factor_red,
        br4_channel_3_count - br4_bkg_para,
        br4_channel_1_count - br4_bkg_perp,
        l1_japan_corr, l2_japan_corr
    )
    
    return {
        'nr_p_all': br4_channel_3_count,
        'nr_s_all': br4_channel_1_count,
        'total_count': br4_total_count,
        'signal_khz': br4_signal_red,
        'r_exp': r_exp,
        'r_scat': r_scat
    }


def calculate_by4_anisotropy(
    burst: List[int],
    routing_channels: np.ndarray,
    all_micro_times: np.ndarray,
    by4_micro_time_min: int,
    by4_micro_time_max: int,
    g_factor_red: float,
    l1_japan_corr: float,
    l2_japan_corr: float,
    by4_bkg_para: float,
    by4_bkg_perp: float,
    micro_res: float
) -> Dict[str, Any]:
    """Calculate BY4 (yellow/delayed) anisotropy data."""
    
    # Find photons in red channels within BY4 (delayed) micro time window
    by4_channel_3_photons = [
        index for index in burst
        if routing_channels[index] == 3 and
        by4_micro_time_min < all_micro_times[index] < by4_micro_time_max
    ]
    
    by4_channel_1_photons = [
        index for index in burst
        if routing_channels[index] == 1 and
        by4_micro_time_min < all_micro_times[index] < by4_micro_time_max
    ]
    
    by4_channel_3_count = len(by4_channel_3_photons)
    by4_channel_1_count = len(by4_channel_1_photons)
    by4_total_count = by4_channel_3_count + by4_channel_1_count
    
    # Calculate duration and signal (using micro_res for proper time conversion)
    by4_duration = (by4_micro_time_max - by4_micro_time_min) * micro_res  # Convert to seconds
    by4_signal_yellow = (by4_total_count / by4_duration) / 1000  # Convert to kHz
    
    # Calculate anisotropy
    r_exp = an.flour_aniso(g_factor_red, by4_channel_3_count, by4_channel_1_count, l1_japan_corr, l2_japan_corr)
    r_scat = an.flour_aniso(
        g_factor_red,
        by4_channel_3_count - by4_bkg_para,
        by4_channel_1_count - by4_bkg_perp,
        l1_japan_corr, l2_japan_corr
    )
    
    return {
        'ny_p_all': by4_channel_3_count,
        'ny_s_all': by4_channel_1_count,
        'total_count': by4_total_count,
        'signal_khz': by4_signal_yellow,
        'r_exp': r_exp,
        'r_scat': r_scat
    }


def calculate_mean_micro_time(
    burst: List[int],
    routing_channels: np.ndarray,
    all_micro_times: np.ndarray,
    bg4_micro_time_min: int,
    bg4_micro_time_max: int,
    micro_res: float
) -> float:
    """Calculate mean micro time for green channel photons (tau)."""
    
    # Get green channel photons in BG4 window
    bg4_channel_0_photons = [
        index for index in burst
        if routing_channels[index] == 0 and
        bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max
    ]
    
    bg4_channel_2_photons = [
        index for index in burst
        if routing_channels[index] == 2 and
        bg4_micro_time_min < all_micro_times[index] < bg4_micro_time_max
    ]
    
    if len(bg4_channel_0_photons) == 0 and len(bg4_channel_2_photons) == 0:
        return np.nan
    
    # Calculate mean micro times for each channel
    micro_times_channel_0 = all_micro_times[bg4_channel_0_photons] if len(bg4_channel_0_photons) > 0 else np.array([])
    micro_times_channel_2 = all_micro_times[bg4_channel_2_photons] if len(bg4_channel_2_photons) > 0 else np.array([])
    
    # Calculate means (if photons exist)
    mean_micro_time_channel_0 = np.mean(micro_times_channel_0) if len(micro_times_channel_0) > 0 else 0
    mean_micro_time_channel_2 = np.mean(micro_times_channel_2) if len(micro_times_channel_2) > 0 else 0
    
    # Combined mean (weighted by the formula from working pipeline)
    combined_mean_micro_time = np.mean([mean_micro_time_channel_0, mean_micro_time_channel_0, mean_micro_time_channel_2])
    
    # Subtract minimum time and convert to ns
    tau_green = (combined_mean_micro_time - bg4_micro_time_min) * micro_res * 1e9
    
    return tau_green


def create_bur_record(
    burst_stats: Dict[str, Any],
    bg4_data: Dict[str, Any], 
    br4_data: Dict[str, Any],
    by4_data: Dict[str, Any]
) -> pd.DataFrame:
    """Create a record for the .bur file."""
    
    return pd.DataFrame({
        'BurstID': [[burst_stats['first_photon'], burst_stats['last_photon']]],
        'First Photon': [burst_stats['first_photon']],
        'Last Photon': [burst_stats['last_photon']],
        'Duration (ms)': [burst_stats['duration']],
        'Mean Macro Time (ms)': [burst_stats['mean_macro_time']],
        'Number of Photons': [burst_stats['num_photons']],
        'Count Rate (kHz)': [burst_stats['count_rate']],
        'Duration (green) (ms)': [burst_stats['duration_green']],
        'Mean Macro Time (green) (ms)': [burst_stats['mean_macro_time_green']],
        'Number of Photons (green)': [burst_stats['num_photons_green']],
        'Green Count Rate (kHz)': [burst_stats['count_rate_green']],
        'Duration (red) (ms)': [burst_stats['duration_red']],
        'Mean Macro Time (red) (ms)': [burst_stats['mean_macro_time_red']],
        'Number of Photons (red)': [burst_stats['num_photons_red']],
        'Red Count Rate (kHz)': [burst_stats['count_rate_red']],
        'Sg (prompt) (kHz)': [bg4_data['signal_khz']],
        'Sr (prompt) (kHz)': [br4_data['signal_khz']],
        'Sy (delay) (kHz)': [by4_data['signal_khz']]
    })


def create_bg4_record(
    burst: List[int],
    bg4_data: Dict[str, Any],
    tau_green: float
) -> pd.DataFrame:
    """Create a record for the .bg4 file."""
    
    return pd.DataFrame({
        'BurstID': [burst],
        'Ng-p-all': [bg4_data['ng_p_all']],
        'Ng-s-all': [bg4_data['ng_s_all']],
        'Number of Photons (fit window) (green)': [bg4_data['total_count']],
        'r Scatter (green)': [bg4_data['r_scat']],
        'r Experimental (green)': [bg4_data['r_exp']],
        'Tau (green)': [tau_green]
    })


def create_br4_record(
    burst: List[int],
    br4_data: Dict[str, Any]
) -> pd.DataFrame:
    """Create a record for the .br4 file."""
    
    return pd.DataFrame({
        'BurstID': [burst],
        'Nr-p-all': [br4_data['nr_p_all']],
        'Nr-s-all': [br4_data['nr_s_all']],
        'Number of Photons (fit window) (red)': [br4_data['total_count']],
        'r Scatter (red)': [br4_data['r_scat']],
        'r Experimental (red)': [br4_data['r_exp']]
    })


def create_by4_record(
    burst: List[int],
    by4_data: Dict[str, Any]
) -> pd.DataFrame:
    """Create a record for the .by4 file."""
    
    return pd.DataFrame({
        'BurstID': [burst],
        'Ny-p-all': [by4_data['ny_p_all']],
        'Ny-s-all': [by4_data['ny_s_all']],
        'Number of Photons (fit window) (yellow)': [by4_data['total_count']],
        'r Scatter (yellow)': [by4_data['r_scat']],
        'r Experimental (yellow)': [by4_data['r_exp']]
    })