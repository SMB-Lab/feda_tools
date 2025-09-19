"""
ThresholdAdjustmentViewModel - Manages threshold calculation and burst detection.

This ViewModel orchestrates the threshold-based burst detection process,
including running average calculation and background noise estimation.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from feda_tools.core import analysis as an
from feda_tools.core import data as dat


class ThresholdAdjustmentViewModel(QObject):
    """ViewModel for orchestrating threshold adjustment and burst detection."""
    
    # Signals for communicating with Views/Coordinator
    calculation_started = pyqtSignal()
    calculation_completed = pyqtSignal()
    calculation_error = pyqtSignal(str)
    
    # Data signals
    running_average_ready = pyqtSignal(object, object)  # xarr, logrunavg
    background_noise_estimated = pyqtSignal(float, float, float)  # mu, std, threshold_value
    burst_indices_ready = pyqtSignal(object)  # burst_index list
    
    def __init__(self):
        super().__init__()
        self.current_config = self._get_default_config()
        
    def configure_analysis(self, config: Dict[str, Any]) -> None:
        """Configure threshold analysis parameters."""
        self.current_config.update(config)
        
    def calculate_running_average(
        self, 
        data_ptu
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the running average of interphoton arrival times.
        
        Args:
            data_ptu: PTU data object
            
        Returns:
            Tuple of (xarr, logrunavg)
        """
        try:
            self.calculation_started.emit()
            
            # Get timing resolutions
            macro_res = data_ptu.get_header().macro_time_resolution
            micro_res = data_ptu.get_header().micro_time_resolution
            
            # Calculate interphoton arrival times
            photon_time_intervals, photon_ids = an.interphoton_arrival_times(
                data_ptu.macro_times, 
                data_ptu.micro_times, 
                macro_res, 
                micro_res
            )
            
            # Calculate running average
            window_size = self.current_config['window_size']
            running_avg = an.running_average(photon_time_intervals, window_size)
            
            # Create x-axis array and log transform
            xarr = np.arange(window_size - 1, len(photon_time_intervals))
            logrunavg = np.log10(running_avg)
            
            # Emit results
            self.running_average_ready.emit(xarr, logrunavg)
            
            return xarr, logrunavg
            
        except Exception as e:
            self.calculation_error.emit(f"Error calculating running average: {str(e)}")
            raise
    
    def estimate_background_noise(
        self, 
        logrunavg: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Estimate background noise parameters using halfnorm fitting.
        
        Args:
            logrunavg: Log running average array
            
        Returns:
            Tuple of (mu, std, threshold_value)
        """
        try:
            bins_y = self.current_config['bins_y']
            
            # Estimate background noise
            mu, std, noise_mean, filtered_logrunavg, bins_logrunavg = an.estimate_background_noise(
                logrunavg, bins_y
            )
            
            # Calculate threshold value
            threshold_multiplier = self.current_config['threshold_multiplier']
            threshold_value = mu - threshold_multiplier * std
            
            # Emit results
            self.background_noise_estimated.emit(mu, std, threshold_value)
            
            return mu, std, threshold_value
            
        except Exception as e:
            self.calculation_error.emit(f"Error estimating background noise: {str(e)}")
            raise
    
    def extract_burst_indices(
        self, 
        logrunavg: np.ndarray, 
        threshold_value: float
    ) -> List[List[int]]:
        """
        Extract burst indices using the calculated threshold.
        
        Args:
            logrunavg: Log running average array
            threshold_value: Threshold for burst detection
            
        Returns:
            List of burst indices
        """
        try:
            # Extract bursts using threshold
            burst_index, filtered_values = dat.extract_greater(logrunavg, threshold_value)
            
            # Emit results
            self.burst_indices_ready.emit(burst_index)
            self.calculation_completed.emit()
            
            return burst_index
            
        except Exception as e:
            self.calculation_error.emit(f"Error extracting burst indices: {str(e)}")
            raise
    
    def run_complete_threshold_analysis(
        self, 
        data_ptu
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, List[List[int]]]:
        """
        Run the complete threshold analysis workflow.
        
        Args:
            data_ptu: PTU data object
            
        Returns:
            Tuple of (xarr, logrunavg, mu, std, threshold_value, burst_index)
        """
        try:
            # Step 1: Calculate running average
            xarr, logrunavg = self.calculate_running_average(data_ptu)
            
            # Step 2: Estimate background noise
            mu, std, threshold_value = self.estimate_background_noise(logrunavg)
            
            # Step 3: Extract burst indices
            burst_index = self.extract_burst_indices(logrunavg, threshold_value)
            
            return xarr, logrunavg, mu, std, threshold_value, burst_index
            
        except Exception as e:
            self.calculation_error.emit(f"Error in complete threshold analysis: {str(e)}")
            raise
    
    def update_threshold_multiplier(
        self, 
        multiplier: float, 
        mu: float, 
        std: float,
        logrunavg: np.ndarray
    ) -> Tuple[float, List[List[int]]]:
        """
        Update the threshold multiplier and recalculate burst indices.
        
        Args:
            multiplier: New threshold multiplier
            mu: Previously calculated mu
            std: Previously calculated std
            logrunavg: Log running average array
            
        Returns:
            Tuple of (new_threshold_value, new_burst_index)
        """
        try:
            # Update config
            self.current_config['threshold_multiplier'] = multiplier
            
            # Recalculate threshold
            threshold_value = mu - multiplier * std
            
            # Extract new burst indices
            burst_index = self.extract_burst_indices(logrunavg, threshold_value)
            
            return threshold_value, burst_index
            
        except Exception as e:
            self.calculation_error.emit(f"Error updating threshold: {str(e)}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'window_size': 30,
            'bins_y': 141,
            'threshold_multiplier': 4.0
        }