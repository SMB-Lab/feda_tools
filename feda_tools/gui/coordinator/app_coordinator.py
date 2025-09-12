"""
AppCoordinator - Central coordination for FEDA Tools GUI.

This coordinator manages data flow between ViewModels and Views using 
QAbstractItemModels. It acts as a "model store" where Views request 
data models, and the coordinator decides which ViewModel to call.
"""

from typing import Dict, Any, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from ..viewmodel import (
    BurstProcessingViewModel,
    DataLoadingViewModel, 
    ThresholdAdjustmentViewModel,
    ResultsViewModel
)
from .burst_data_model import BurstDataModel
from .anisotropy_data_model import AnisotropyDataModel
from .processing_state_model import ProcessingStateModel


class AppCoordinator(QObject):
    """
    Central coordinator for managing data flow in the FEDA Tools GUI.
    
    The coordinator:
    - Manages ViewModels and their interactions
    - Provides QAbstractItemModels to Views
    - Handles inter-view communication
    - Maintains application state
    """
    
    # Global application signals
    application_state_changed = pyqtSignal(str)  # State description
    error_occurred = pyqtSignal(str, str)  # Error type, error message
    progress_updated = pyqtSignal(int)  # Overall progress percentage
    
    def __init__(self):
        super().__init__()
        
        # Initialize ViewModels
        self.burst_processing_vm = BurstProcessingViewModel()
        self.data_loading_vm = DataLoadingViewModel()
        self.threshold_adjustment_vm = ThresholdAdjustmentViewModel()
        self.results_vm = ResultsViewModel()
        
        # Initialize Data Models
        self.burst_data_model = BurstDataModel()
        self.anisotropy_data_model = AnisotropyDataModel()
        self.processing_state_model = ProcessingStateModel()
        
        # Application state
        self.current_data_ptu = None
        self.current_data_irf = None
        self.current_data_bkg = None
        self.current_burst_index = None
        self.current_file_path = None
        
        # Connect ViewModel signals
        self._connect_viewmodel_signals()
        
        # Set initial state
        self.application_state_changed.emit("Ready")
    
    def _connect_viewmodel_signals(self):
        """Connect signals from all ViewModels."""
        
        # Data Loading ViewModel signals
        self.data_loading_vm.data_loaded.connect(self._on_data_loaded)
        self.data_loading_vm.loading_error.connect(
            lambda msg: self.error_occurred.emit("Data Loading", msg)
        )
        
        # Threshold Adjustment ViewModel signals
        self.threshold_adjustment_vm.burst_indices_ready.connect(self._on_burst_indices_ready)
        self.threshold_adjustment_vm.calculation_error.connect(
            lambda msg: self.error_occurred.emit("Threshold Calculation", msg)
        )
        
        # Burst Processing ViewModel signals
        self.burst_processing_vm.data_ready.connect(self._on_burst_data_ready)
        self.burst_processing_vm.processing_error.connect(
            lambda msg: self.error_occurred.emit("Burst Processing", msg)
        )
        self.burst_processing_vm.processing_progress.connect(self.progress_updated.emit)
        
        # Results ViewModel signals
        self.results_vm.save_completed.connect(
            lambda dir: self.application_state_changed.emit(f"Results saved to {dir}")
        )
        self.results_vm.save_error.connect(
            lambda msg: self.error_occurred.emit("File Save", msg)
        )
    
    # ========================================================================
    # Public API for Views
    # ========================================================================
    
    def get_burst_data_model(self) -> BurstDataModel:
        """Get the burst data model for Views."""
        return self.burst_data_model
    
    def get_anisotropy_data_model(self) -> AnisotropyDataModel:
        """Get the anisotropy data model for Views."""
        return self.anisotropy_data_model
    
    def get_processing_state_model(self) -> ProcessingStateModel:
        """Get the processing state model for Views."""
        return self.processing_state_model
    
    # ========================================================================
    # Data Loading Operations
    # ========================================================================
    
    def load_ptu_file(self, file_path: str, irf_path: Optional[str] = None, 
                      bkg_path: Optional[str] = None):
        """
        Load PTU data files.
        
        Args:
            file_path: Main PTU file path
            irf_path: Optional IRF file path
            bkg_path: Optional background file path
        """
        self.current_file_path = file_path
        self.application_state_changed.emit("Loading PTU data...")
        
        self.data_loading_vm.set_file_paths(file_path, irf_path, bkg_path)
        self.data_loading_vm.load_ptu_data()
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded files."""
        return self.data_loading_vm.get_file_info()
    
    # ========================================================================
    # Threshold Adjustment Operations
    # ========================================================================
    
    def run_threshold_analysis(self, config: Optional[Dict[str, Any]] = None):
        """
        Run threshold-based burst detection.
        
        Args:
            config: Optional configuration parameters
        """
        if not self.current_data_ptu:
            self.error_occurred.emit("Threshold Analysis", "No PTU data loaded")
            return
        
        self.application_state_changed.emit("Running threshold analysis...")
        
        if config:
            self.threshold_adjustment_vm.configure_analysis(config)
        
        self.threshold_adjustment_vm.run_complete_threshold_analysis(self.current_data_ptu)
    
    def update_threshold_multiplier(self, multiplier: float):
        """
        Update threshold multiplier and recalculate bursts.
        
        Args:
            multiplier: New threshold multiplier
        """
        if hasattr(self, '_last_mu') and hasattr(self, '_last_std') and hasattr(self, '_last_logrunavg'):
            self.threshold_adjustment_vm.update_threshold_multiplier(
                multiplier, self._last_mu, self._last_std, self._last_logrunavg
            )
    
    # ========================================================================
    # Burst Processing Operations
    # ========================================================================
    
    def run_burst_processing(self, config: Optional[Dict[str, Any]] = None):
        """
        Run burst processing analysis.
        
        Args:
            config: Optional configuration parameters
        """
        if not self.current_data_ptu or not self.current_burst_index:
            self.error_occurred.emit("Burst Processing", "No data or burst indices available")
            return
        
        self.application_state_changed.emit("Processing bursts...")
        
        if config:
            self.burst_processing_vm.configure_analysis(config)
        
        self.burst_processing_vm.run_burst_analysis(
            self.current_burst_index, 
            self.current_data_ptu
        )
    
    # ========================================================================
    # Results Operations
    # ========================================================================
    
    def save_results(self, output_directory: str):
        """
        Save analysis results to files.
        
        Args:
            output_directory: Directory for output files
        """
        if not self.burst_data_model.has_data():
            self.error_occurred.emit("Save Results", "No results to save")
            return
        
        self.results_vm.set_output_directory(output_directory)
        
        # Get DataFrames from models
        bi4_bur_df = self.burst_data_model.get_dataframe()
        bg4_df, br4_df, by4_df = self.anisotropy_data_model.get_dataframes()
        
        self.results_vm.save_analysis_results(
            self.current_file_path,
            bi4_bur_df, bg4_df, br4_df, by4_df,
            output_directory
        )
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of analysis results."""
        if not self.burst_data_model.has_data():
            return {}
        
        # Get DataFrames from models
        bi4_bur_df = self.burst_data_model.get_dataframe()
        bg4_df, br4_df, by4_df = self.anisotropy_data_model.get_dataframes()
        
        return self.results_vm.get_result_summary(bi4_bur_df, bg4_df, br4_df, by4_df)
    
    # ========================================================================
    # Private Event Handlers
    # ========================================================================
    
    def _on_data_loaded(self, data_ptu, data_irf, data_bkg):
        """Handle data loading completion."""
        self.current_data_ptu = data_ptu
        self.current_data_irf = data_irf
        self.current_data_bkg = data_bkg
        
        # Update processing state
        self.processing_state_model.set_data_loaded(True)
        
        self.application_state_changed.emit("PTU data loaded successfully")
    
    def _on_burst_indices_ready(self, burst_index):
        """Handle burst indices calculation completion."""
        self.current_burst_index = burst_index
        
        # Update processing state
        self.processing_state_model.set_threshold_calculated(True)
        self.processing_state_model.set_burst_count(len(burst_index))
        
        self.application_state_changed.emit(f"Found {len(burst_index)} potential bursts")
    
    def _on_burst_data_ready(self, bi4_bur_df, bg4_df, br4_df, by4_df):
        """Handle burst processing completion."""
        
        # Update data models
        self.burst_data_model.set_data(bi4_bur_df)
        self.anisotropy_data_model.set_data(bg4_df, br4_df, by4_df)
        
        # Update processing state
        self.processing_state_model.set_processing_complete(True)
        self.processing_state_model.set_processed_burst_count(len(bi4_bur_df))
        
        self.application_state_changed.emit(f"Processed {len(bi4_bur_df)} bursts successfully")
    
    # ========================================================================
    # Configuration Management
    # ========================================================================
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all analysis steps."""
        return {
            'threshold': self.threshold_adjustment_vm._get_default_config(),
            'burst_processing': self.burst_processing_vm._get_default_config()
        }
    
    def apply_config(self, config: Dict[str, Any]):
        """
        Apply configuration to all ViewModels.
        
        Args:
            config: Configuration dictionary with sections for each ViewModel
        """
        if 'threshold' in config:
            self.threshold_adjustment_vm.configure_analysis(config['threshold'])
        
        if 'burst_processing' in config:
            self.burst_processing_vm.configure_analysis(config['burst_processing'])