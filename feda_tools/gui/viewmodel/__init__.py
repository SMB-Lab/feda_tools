"""
ViewModel package for FEDA Tools GUI.

This package contains ViewModels that orchestrate between the View layer and 
core business logic. ViewModels are thin orchestration layers that:

1. Call core functions from feda_tools.core
2. Prepare data for QAbstractItemModels
3. Handle error management and user feedback
4. Emit signals for communication with the App Coordinator

ViewModels should be ~50-100 lines and contain no business logic themselves.
"""

from .burst_processing_viewmodel import BurstProcessingViewModel
from .data_loading_viewmodel import DataLoadingViewModel
from .threshold_adjustment_viewmodel import ThresholdAdjustmentViewModel
from .results_viewmodel import ResultsViewModel

__all__ = [
    'BurstProcessingViewModel',
    'DataLoadingViewModel', 
    'ThresholdAdjustmentViewModel',
    'ResultsViewModel'
]