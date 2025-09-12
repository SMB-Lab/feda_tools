"""
ProcessingStateModel - QAbstractItemModel for pipeline progress/status.

This model tracks the state of the analysis pipeline and provides
status information for Views to display progress and current state.
"""

from typing import Any, Dict
from PyQt6.QtCore import QAbstractListModel, QModelIndex, Qt, QVariant, pyqtSignal


class ProcessingStateModel(QAbstractListModel):
    """
    Qt model for tracking analysis pipeline state and progress.
    
    This model maintains information about:
    - Which analysis steps are complete
    - Progress of current operations  
    - Data loading status
    - Burst processing statistics
    """
    
    # Signal emitted when state changes
    state_changed = pyqtSignal(str, bool)  # state_name, is_complete
    
    def __init__(self):
        super().__init__()
        
        # Processing state tracking
        self._state = {
            'data_loaded': False,
            'threshold_calculated': False,
            'processing_complete': False
        }
        
        # Statistics tracking
        self._stats = {
            'file_path': '',
            'total_events': 0,
            'potential_bursts': 0,
            'processed_bursts': 0,
            'threshold_value': 0.0,
            'processing_progress': 0
        }
        
        # State descriptions for display
        self._state_descriptions = [
            ('Data Loading', 'data_loaded'),
            ('Threshold Calculation', 'threshold_calculated'), 
            ('Burst Processing', 'processing_complete')
        ]
    
    # ========================================================================
    # State Management
    # ========================================================================
    
    def set_data_loaded(self, loaded: bool, file_path: str = '', total_events: int = 0):
        """
        Set data loading state.
        
        Args:
            loaded: Whether data is loaded
            file_path: Path to loaded file
            total_events: Number of events in file
        """
        self._state['data_loaded'] = loaded
        self._stats['file_path'] = file_path
        self._stats['total_events'] = total_events
        
        self.state_changed.emit('data_loaded', loaded)
        self._emit_data_changed()
    
    def set_threshold_calculated(self, calculated: bool, threshold_value: float = 0.0):
        """
        Set threshold calculation state.
        
        Args:
            calculated: Whether threshold is calculated
            threshold_value: Calculated threshold value
        """
        self._state['threshold_calculated'] = calculated
        self._stats['threshold_value'] = threshold_value
        
        self.state_changed.emit('threshold_calculated', calculated)
        self._emit_data_changed()
    
    def set_processing_complete(self, complete: bool):
        """
        Set processing completion state.
        
        Args:
            complete: Whether processing is complete
        """
        self._state['processing_complete'] = complete
        if complete:
            self._stats['processing_progress'] = 100
        
        self.state_changed.emit('processing_complete', complete)
        self._emit_data_changed()
    
    def set_burst_count(self, count: int):
        """
        Set the number of potential bursts found.
        
        Args:
            count: Number of potential bursts
        """
        self._stats['potential_bursts'] = count
        self._emit_data_changed()
    
    def set_processed_burst_count(self, count: int):
        """
        Set the number of processed bursts.
        
        Args:
            count: Number of processed bursts
        """
        self._stats['processed_bursts'] = count
        self._emit_data_changed()
    
    def set_processing_progress(self, progress: int):
        """
        Set processing progress percentage.
        
        Args:
            progress: Progress percentage (0-100)
        """
        self._stats['processing_progress'] = max(0, min(100, progress))
        self._emit_data_changed()
    
    # ========================================================================
    # State Queries
    # ========================================================================
    
    def is_data_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._state['data_loaded']
    
    def is_threshold_calculated(self) -> bool:
        """Check if threshold is calculated."""
        return self._state['threshold_calculated']
    
    def is_processing_complete(self) -> bool:
        """Check if processing is complete."""
        return self._state['processing_complete']
    
    def can_run_threshold_analysis(self) -> bool:
        """Check if threshold analysis can be run."""
        return self._state['data_loaded']
    
    def can_run_burst_processing(self) -> bool:
        """Check if burst processing can be run."""
        return self._state['data_loaded'] and self._state['threshold_calculated']
    
    def can_save_results(self) -> bool:
        """Check if results can be saved."""
        return self._state['processing_complete']
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state and statistics."""
        return {
            'state': self._state.copy(),
            'stats': self._stats.copy(),
            'can_run_threshold': self.can_run_threshold_analysis(),
            'can_run_processing': self.can_run_burst_processing(),
            'can_save_results': self.can_save_results()
        }
    
    def get_progress_percentage(self) -> int:
        """Get overall progress percentage."""
        if self._state['processing_complete']:
            return 100
        elif self._state['threshold_calculated']:
            return 60 + int(self._stats['processing_progress'] * 0.4)
        elif self._state['data_loaded']:
            return 30
        else:
            return 0
    
    # ========================================================================
    # QAbstractListModel Interface
    # ========================================================================
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of processing steps."""
        if parent.isValid():
            return 0
        return len(self._state_descriptions)
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return data for the given index and role."""
        if not index.isValid():
            return QVariant()
        
        row = index.row()
        if row >= len(self._state_descriptions):
            return QVariant()
        
        description, state_key = self._state_descriptions[row]
        is_complete = self._state[state_key]
        
        if role == Qt.ItemDataRole.DisplayRole:
            status = "✓" if is_complete else "○"
            return f"{status} {description}"
        
        elif role == Qt.ItemDataRole.ToolTipRole:
            if state_key == 'data_loaded':
                if is_complete:
                    return f"Loaded: {self._stats['file_path']} ({self._stats['total_events']} events)"
                else:
                    return "No data loaded"
            elif state_key == 'threshold_calculated':
                if is_complete:
                    return f"Threshold: {self._stats['threshold_value']:.3f} ({self._stats['potential_bursts']} bursts found)"
                else:
                    return "Threshold not calculated"
            elif state_key == 'processing_complete':
                if is_complete:
                    return f"Processed {self._stats['processed_bursts']} bursts successfully"
                else:
                    return f"Processing: {self._stats['processing_progress']}%"
        
        elif role == Qt.ItemDataRole.UserRole:
            # Return state information for custom roles
            return {
                'state_key': state_key,
                'is_complete': is_complete,
                'description': description
            }
        
        return QVariant()
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return item flags."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _emit_data_changed(self):
        """Emit data changed signal for all items."""
        if self.rowCount() > 0:
            top_left = self.index(0)
            bottom_right = self.index(self.rowCount() - 1)
            self.dataChanged.emit(top_left, bottom_right)
    
    def reset_state(self):
        """Reset all state to initial values."""
        self.beginResetModel()
        
        self._state = {
            'data_loaded': False,
            'threshold_calculated': False,
            'processing_complete': False
        }
        
        self._stats = {
            'file_path': '',
            'total_events': 0,
            'potential_bursts': 0,
            'processed_bursts': 0,
            'threshold_value': 0.0,
            'processing_progress': 0
        }
        
        self.endResetModel()
    
    def get_status_text(self) -> str:
        """Get a human-readable status text."""
        if self._state['processing_complete']:
            return f"Complete - {self._stats['processed_bursts']} bursts processed"
        elif self._state['threshold_calculated']:
            return f"Processing - {self._stats['processing_progress']}%"
        elif self._state['data_loaded']:
            return f"Ready - {self._stats['total_events']} events loaded"
        else:
            return "No data loaded"