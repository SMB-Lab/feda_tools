"""
Coordinator package for FEDA Tools GUI.

This package contains the App Coordinator and QAbstractItemModels that
manage data flow between ViewModels and Views. The coordinator acts as
a central "model store" where:

1. Views request QAbstractItemModels from the coordinator
2. Coordinator calls appropriate ViewModels to populate models
3. All inter-view communication goes through the coordinator
4. Models automatically notify Views of data changes via Qt signals

Architecture:
- AppCoordinator: Central coordination class
- BurstDataModel: QAbstractItemModel for .bur file data
- AnisotropyDataModel: QAbstractItemModel for .bg4/.br4/.by4 data
- ProcessingStateModel: QAbstractItemModel for pipeline status
"""

from .app_coordinator import AppCoordinator
from .burst_data_model import BurstDataModel
from .anisotropy_data_model import AnisotropyDataModel
from .processing_state_model import ProcessingStateModel

__all__ = [
    'AppCoordinator',
    'BurstDataModel',
    'AnisotropyDataModel', 
    'ProcessingStateModel'
]