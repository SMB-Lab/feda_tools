"""
BurstProcessingViewModel - Orchestrates core burst analysis functions.

This ViewModel is responsible for coordinating between the View layer and the 
core burst processing logic. It prepares data for QAbstractItemModels and 
manages the burst analysis workflow.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from feda_tools.core import burst_processing as bp
from feda_tools.core import analysis as an
from feda_tools.core import data as dat


class BurstProcessingViewModel(QObject):
    """ViewModel for orchestrating burst processing operations."""
    
    # Signals for communicating with Views/Coordinator
    processing_started = pyqtSignal()
    processing_progress = pyqtSignal(int)  # Progress percentage
    processing_completed = pyqtSignal()
    processing_error = pyqtSignal(str)
    data_ready = pyqtSignal(object, object, object, object)  # bi4_bur_df, bg4_df, br4_df, by4_df
    
    def __init__(self):
        super().__init__()
        self.current_config = {}
        
    def configure_analysis(self, config: Dict[str, Any]) -> None:
        """Configure analysis parameters."""
        self.current_config = config
        
    def run_burst_analysis(
        self, 
        burst_index: List[List[int]],
        data_ptu,
        progress_callback=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the burst analysis using the configured parameters.
        
        Args:
            burst_index: List of burst indices
            data_ptu: PTU data object
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (bi4_bur_df, bg4_df, br4_df, by4_df)
        """
        try:
            self.processing_started.emit()
            
            # Extract configuration parameters
            config = self._get_default_config()
            config.update(self.current_config)
            
            # Get timing resolutions
            macro_res = data_ptu.get_header().macro_time_resolution
            micro_res = data_ptu.get_header().micro_time_resolution
            
            # Process bursts using core system
            bi4_bur_df, bg4_df, br4_df, by4_df = bp.process_bursts(
                burst_index=burst_index,
                all_macro_times=data_ptu.macro_times,
                all_micro_times=data_ptu.micro_times,
                routing_channels=data_ptu.routing_channels,
                macro_res=macro_res,
                micro_res=micro_res,
                **config
            )
            
            # Emit data ready signal
            self.data_ready.emit(bi4_bur_df, bg4_df, br4_df, by4_df)
            self.processing_completed.emit()
            
            return bi4_bur_df, bg4_df, br4_df, by4_df
            
        except Exception as e:
            self.processing_error.emit(str(e))
            raise
    
    def calculate_derived_metrics(
        self, 
        bi4_bur_df: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """
        Calculate derived metrics for plotting (e.g., Sg/Sr ratios).
        
        Args:
            bi4_bur_df: Burst DataFrame
            
        Returns:
            Dictionary of calculated metrics for plotting
        """
        try:
            metrics = {
                'sg_sr_ratios': [],
                'mean_macro_times': [],
                's_prompt_s_total': []
            }
            
            if len(bi4_bur_df) == 0:
                return metrics
            
            # Calculate Sg/Sr ratios
            sg_values = bi4_bur_df['Sg (prompt) (kHz)'].values
            sr_values = bi4_bur_df['Sr (prompt) (kHz)'].values
            sy_values = bi4_bur_df['Sy (delay) (kHz)'].values
            
            for i, (sg, sr, sy) in enumerate(zip(sg_values, sr_values, sy_values)):
                # Sg/Sr ratio (handle division by zero)
                if sr > 0:
                    sg_sr_ratio = sg / sr
                    metrics['sg_sr_ratios'].append(sg_sr_ratio)
                    metrics['mean_macro_times'].append(bi4_bur_df['Mean Macro Time (ms)'].iloc[i])
                
                # S(prompt)/S(total) ratio
                total_signal = sg + sr + sy
                if total_signal > 0:
                    prompt_ratio = (sg + sr) / total_signal
                    metrics['s_prompt_s_total'].append(prompt_ratio)
            
            return metrics
            
        except Exception as e:
            self.processing_error.emit(f"Error calculating metrics: {str(e)}")
            return {'sg_sr_ratios': [], 'mean_macro_times': [], 's_prompt_s_total': []}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'min_photon_count': 60,
            'min_photon_count_green': 20,
            'bg4_micro_time_min': 1000,
            'bg4_micro_time_max': 7000,
            'br4_micro_time_min': 1000,
            'br4_micro_time_max': 7000,
            'by4_micro_time_min': 13500,
            'by4_micro_time_max': 18000,
            'g_factor': 1.04,
            'g_factor_red': 2.5,
            'l1_japan_corr': 0.0308,
            'l2_japan_corr': 0.0368,
            'bg4_bkg_para': 0,
            'bg4_bkg_perp': 0,
            'br4_bkg_para': 0,
            'br4_bkg_perp': 0,
            'by4_bkg_para': 0,
            'by4_bkg_perp': 0
        }