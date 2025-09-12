"""
View package for FEDA Tools GUI.

This package contains all View components (GUI widgets) that display data
and interact with users. Views in the MVVM architecture:

1. Receive QAbstractItemModels from the App Coordinator
2. Display data using Qt's model/view framework
3. Emit signals to the Coordinator (never call other Views directly)
4. Handle user interactions and UI logic only

Views should not contain business logic or call core functions directly.
All data operations should go through ViewModels via the Coordinator.
"""

from .threshold_adjustment import ThresholdAdjustmentWidget
from .process_analysis import ProcessAnalysisWidget
from .widgets import PlotWidget
from .state_handler import StateHandler

__all__ = [
    'ThresholdAdjustmentWidget',
    'ProcessAnalysisWidget',
    'PlotWidget',
    'StateHandler'
]