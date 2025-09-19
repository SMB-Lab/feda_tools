"""
BurstDataModel - QAbstractItemModel for burst analysis results.

This model wraps burst analysis DataFrames for use with Qt's model/view system.
It provides access to .bur file data (burst statistics).
"""

from typing import Any, Optional
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant


class BurstDataModel(QAbstractTableModel):
    """
    Qt model for burst analysis results (.bur data).
    
    This model provides a Qt interface to burst statistics DataFrames,
    allowing Views to display and interact with the data using Qt's 
    model/view framework.
    """
    
    def __init__(self):
        super().__init__()
        self._dataframe = pd.DataFrame()
        self._column_names = []
    
    def set_data(self, dataframe: pd.DataFrame):
        """
        Set the burst data DataFrame.
        
        Args:
            dataframe: Burst statistics DataFrame (bi4_bur_df)
        """
        self.beginResetModel()
        self._dataframe = dataframe.copy() if dataframe is not None else pd.DataFrame()
        self._column_names = list(self._dataframe.columns) if not self._dataframe.empty else []
        self.endResetModel()
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._dataframe.copy()
    
    def has_data(self) -> bool:
        """Check if the model contains data."""
        return not self._dataframe.empty
    
    # ========================================================================
    # QAbstractTableModel Interface
    # ========================================================================
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows."""
        if parent.isValid():
            return 0
        return len(self._dataframe)
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns."""
        if parent.isValid():
            return 0
        return len(self._column_names)
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return data for the given index and role."""
        if not index.isValid() or self._dataframe.empty:
            return QVariant()
        
        row = index.row()
        col = index.column()
        
        if row >= len(self._dataframe) or col >= len(self._column_names):
            return QVariant()
        
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._dataframe.iloc[row, col]
            
            # Format different data types appropriately
            if pd.isna(value):
                return "N/A"
            elif isinstance(value, float):
                # Format floats to reasonable precision
                return f"{value:.6f}"
            elif isinstance(value, (list, tuple)):
                # Handle BurstID arrays
                return str(value)
            else:
                return str(value)
        
        elif role == Qt.ItemDataRole.ToolTipRole:
            # Provide tooltips with column descriptions
            column_name = self._column_names[col]
            return self._get_column_tooltip(column_name)
        
        return QVariant()
    
    def headerData(self, section: int, orientation: Qt.Orientation, 
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return header data."""
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._column_names):
                return self._column_names[section]
        elif orientation == Qt.Orientation.Vertical:
            if 0 <= section < len(self._dataframe):
                return str(section + 1)  # 1-based row numbers
        
        return QVariant()
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return item flags."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    
    # ========================================================================
    # Data Access Methods
    # ========================================================================
    
    def get_burst_by_row(self, row: int) -> Optional[pd.Series]:
        """
        Get burst data for a specific row.
        
        Args:
            row: Row index
            
        Returns:
            Burst data as pandas Series, or None if invalid row
        """
        if 0 <= row < len(self._dataframe):
            return self._dataframe.iloc[row]
        return None
    
    def get_column_data(self, column_name: str) -> pd.Series:
        """
        Get all data for a specific column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Column data as pandas Series
        """
        if column_name in self._dataframe.columns:
            return self._dataframe[column_name]
        return pd.Series(dtype=float)
    
    def get_plotting_data(self, x_column: str, y_column: str) -> tuple:
        """
        Get data for plotting two columns.
        
        Args:
            x_column: X-axis column name
            y_column: Y-axis column name
            
        Returns:
            Tuple of (x_data, y_data) as numpy arrays
        """
        if x_column in self._dataframe.columns and y_column in self._dataframe.columns:
            x_data = self._dataframe[x_column].values
            y_data = self._dataframe[y_column].values
            
            # Remove any NaN values
            valid_mask = ~(pd.isna(x_data) | pd.isna(y_data))
            return x_data[valid_mask], y_data[valid_mask]
        
        return [], []
    
    def get_statistics_summary(self) -> dict:
        """
        Get basic statistics for numerical columns.
        
        Returns:
            Dictionary of column statistics
        """
        if self._dataframe.empty:
            return {}
        
        summary = {}
        numerical_cols = self._dataframe.select_dtypes(include=['number']).columns
        
        for col in numerical_cols:
            if col != 'BurstID':  # Skip BurstID which is typically a list
                data = self._dataframe[col].dropna()
                if len(data) > 0:
                    summary[col] = {
                        'count': len(data),
                        'mean': data.mean(),
                        'std': data.std(),
                        'min': data.min(),
                        'max': data.max()
                    }
        
        return summary
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_column_tooltip(self, column_name: str) -> str:
        """Get tooltip description for a column."""
        tooltips = {
            'BurstID': 'List of photon indices that make up this burst',
            'First Photon': 'Index of the first photon in the burst',
            'Last Photon': 'Index of the last photon in the burst',
            'Duration (ms)': 'Duration of the burst in milliseconds',
            'Mean Macro Time (ms)': 'Average macro time of all photons in the burst',
            'Number of Photons': 'Total number of photons in the burst',
            'Count Rate (kHz)': 'Photon count rate for the burst in kHz',
            'Number of Photons (green)': 'Number of green channel photons',
            'Number of Photons (red)': 'Number of red channel photons',
            'Green Count Rate (kHz)': 'Green channel count rate in kHz',
            'Red Count Rate (kHz)': 'Red channel count rate in kHz',
            'Sg (prompt) (kHz)': 'Green channel signal in prompt window (kHz)',
            'Sr (prompt) (kHz)': 'Red channel signal in prompt window (kHz)',
            'Sy (delay) (kHz)': 'Signal in delay window (yellow) (kHz)'
        }
        
        return tooltips.get(column_name, f"Data for {column_name}")