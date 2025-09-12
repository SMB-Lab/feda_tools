"""
AnisotropyDataModel - QAbstractItemModel for anisotropy analysis results.

This model wraps anisotropy DataFrames (BG4, BR4, BY4) for use with Qt's 
model/view system. It provides unified access to all anisotropy data.
"""

from typing import Any, Optional, Tuple
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant


class AnisotropyDataModel(QAbstractTableModel):
    """
    Qt model for anisotropy analysis results (BG4/BR4/BY4 data).
    
    This model provides a unified interface to all three anisotropy datasets,
    allowing Views to display and analyze anisotropy data across different
    spectral channels and time windows.
    """
    
    def __init__(self):
        super().__init__()
        self._bg4_df = pd.DataFrame()  # Green channel (prompt)
        self._br4_df = pd.DataFrame()  # Red channel (prompt) 
        self._by4_df = pd.DataFrame()  # Yellow channel (delay)
        self._current_dataset = 'bg4'  # Currently displayed dataset
        self._datasets = {
            'bg4': 'Green (BG4)',
            'br4': 'Red (BR4)', 
            'by4': 'Yellow (BY4)'
        }
    
    def set_data(self, bg4_df: pd.DataFrame, br4_df: pd.DataFrame, by4_df: pd.DataFrame):
        """
        Set all anisotropy DataFrames.
        
        Args:
            bg4_df: Green channel anisotropy DataFrame
            br4_df: Red channel anisotropy DataFrame  
            by4_df: Yellow channel anisotropy DataFrame
        """
        self.beginResetModel()
        self._bg4_df = bg4_df.copy() if bg4_df is not None else pd.DataFrame()
        self._br4_df = br4_df.copy() if br4_df is not None else pd.DataFrame()
        self._by4_df = by4_df.copy() if by4_df is not None else pd.DataFrame()
        self.endResetModel()
    
    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get all underlying DataFrames."""
        return self._bg4_df.copy(), self._br4_df.copy(), self._by4_df.copy()
    
    def set_current_dataset(self, dataset: str):
        """
        Switch the currently displayed dataset.
        
        Args:
            dataset: Dataset name ('bg4', 'br4', or 'by4')
        """
        if dataset in self._datasets:
            self.beginResetModel()
            self._current_dataset = dataset
            self.endResetModel()
    
    def get_current_dataset(self) -> str:
        """Get the currently displayed dataset name."""
        return self._current_dataset
    
    def get_available_datasets(self) -> dict:
        """Get available datasets with their display names."""
        return self._datasets.copy()
    
    def has_data(self) -> bool:
        """Check if any dataset contains data."""
        return not (self._bg4_df.empty and self._br4_df.empty and self._by4_df.empty)
    
    def _get_current_dataframe(self) -> pd.DataFrame:
        """Get the currently selected DataFrame."""
        if self._current_dataset == 'bg4':
            return self._bg4_df
        elif self._current_dataset == 'br4':
            return self._br4_df
        elif self._current_dataset == 'by4':
            return self._by4_df
        return pd.DataFrame()
    
    # ========================================================================
    # QAbstractTableModel Interface
    # ========================================================================
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows in the current dataset."""
        if parent.isValid():
            return 0
        return len(self._get_current_dataframe())
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns in the current dataset."""
        if parent.isValid():
            return 0
        df = self._get_current_dataframe()
        return len(df.columns) if not df.empty else 0
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return data for the given index and role."""
        if not index.isValid():
            return QVariant()
        
        df = self._get_current_dataframe()
        if df.empty:
            return QVariant()
        
        row = index.row()
        col = index.column()
        
        if row >= len(df) or col >= len(df.columns):
            return QVariant()
        
        if role == Qt.ItemDataRole.DisplayRole:
            value = df.iloc[row, col]
            
            # Format different data types appropriately
            if pd.isna(value):
                return "N/A"
            elif isinstance(value, float):
                # Format floats to reasonable precision
                return f"{value:.6f}"
            elif isinstance(value, (list, tuple)):
                # Handle BurstID arrays (show length instead of full array)
                return f"Burst[{len(value)} photons]"
            else:
                return str(value)
        
        elif role == Qt.ItemDataRole.ToolTipRole:
            # Provide tooltips with column descriptions
            df = self._get_current_dataframe()
            if not df.empty and col < len(df.columns):
                column_name = df.columns[col]
                return self._get_column_tooltip(column_name, self._current_dataset)
        
        return QVariant()
    
    def headerData(self, section: int, orientation: Qt.Orientation, 
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return header data."""
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        
        df = self._get_current_dataframe()
        if df.empty:
            return QVariant()
        
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(df.columns):
                return df.columns[section]
        elif orientation == Qt.Orientation.Vertical:
            if 0 <= section < len(df):
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
    
    def get_anisotropy_data(self, dataset: str, column: str) -> pd.Series:
        """
        Get anisotropy data for a specific dataset and column.
        
        Args:
            dataset: Dataset name ('bg4', 'br4', or 'by4')
            column: Column name
            
        Returns:
            Column data as pandas Series
        """
        df = getattr(self, f'_{dataset}_df', pd.DataFrame())
        if column in df.columns:
            return df[column]
        return pd.Series(dtype=float)
    
    def get_r_values(self, dataset: str, r_type: str = 'experimental') -> pd.Series:
        """
        Get anisotropy r values for a dataset.
        
        Args:
            dataset: Dataset name ('bg4', 'br4', or 'by4')
            r_type: Type of r value ('experimental' or 'scatter')
            
        Returns:
            R values as pandas Series
        """
        if r_type == 'experimental':
            column_map = {
                'bg4': 'r Experimental (green)',
                'br4': 'r Experimental (red)',
                'by4': 'r Experimental (yellow)'
            }
        else:  # scatter
            column_map = {
                'bg4': 'r Scatter (green)',
                'br4': 'r Scatter (red)', 
                'by4': 'r Scatter (yellow)'
            }
        
        column_name = column_map.get(dataset)
        if column_name:
            return self.get_anisotropy_data(dataset, column_name)
        
        return pd.Series(dtype=float)
    
    def get_photon_counts(self, dataset: str) -> dict:
        """
        Get photon count data for a dataset.
        
        Args:
            dataset: Dataset name ('bg4', 'br4', or 'by4')
            
        Returns:
            Dictionary with parallel, perpendicular, and total counts
        """
        if dataset == 'bg4':
            return {
                'parallel': self.get_anisotropy_data(dataset, 'Ng-p-all'),
                'perpendicular': self.get_anisotropy_data(dataset, 'Ng-s-all'),
                'total': self.get_anisotropy_data(dataset, 'Number of Photons (fit window) (green)')
            }
        elif dataset == 'br4':
            return {
                'parallel': self.get_anisotropy_data(dataset, 'Nr-p-all'),
                'perpendicular': self.get_anisotropy_data(dataset, 'Nr-s-all'),
                'total': self.get_anisotropy_data(dataset, 'Number of Photons (fit window) (red)')
            }
        elif dataset == 'by4':
            return {
                'parallel': self.get_anisotropy_data(dataset, 'Ny-p-all'),
                'perpendicular': self.get_anisotropy_data(dataset, 'Ny-s-all'),
                'total': self.get_anisotropy_data(dataset, 'Number of Photons (fit window) (yellow)')
            }
        
        return {'parallel': pd.Series(), 'perpendicular': pd.Series(), 'total': pd.Series()}
    
    def get_tau_values(self) -> pd.Series:
        """Get tau (lifetime) values from the green dataset."""
        return self.get_anisotropy_data('bg4', 'Tau (green)')
    
    def get_combined_statistics(self) -> dict:
        """
        Get combined statistics across all datasets.
        
        Returns:
            Dictionary of statistics for each dataset
        """
        stats = {}
        
        for dataset_key, dataset_name in self._datasets.items():
            df = getattr(self, f'_{dataset_key}_df')
            if not df.empty:
                # Get r values
                r_exp = self.get_r_values(dataset_key, 'experimental')
                r_scat = self.get_r_values(dataset_key, 'scatter')
                
                # Get photon counts
                counts = self.get_photon_counts(dataset_key)
                
                stats[dataset_name] = {
                    'burst_count': len(df),
                    'mean_r_exp': r_exp.mean() if len(r_exp) > 0 else 0,
                    'std_r_exp': r_exp.std() if len(r_exp) > 0 else 0,
                    'mean_r_scat': r_scat.mean() if len(r_scat) > 0 else 0,
                    'std_r_scat': r_scat.std() if len(r_scat) > 0 else 0,
                    'mean_total_photons': counts['total'].mean() if len(counts['total']) > 0 else 0
                }
                
                # Add tau statistics for green dataset
                if dataset_key == 'bg4':
                    tau_values = self.get_tau_values()
                    stats[dataset_name].update({
                        'mean_tau': tau_values.mean() if len(tau_values) > 0 else 0,
                        'std_tau': tau_values.std() if len(tau_values) > 0 else 0
                    })
        
        return stats
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_column_tooltip(self, column_name: str, dataset: str) -> str:
        """Get tooltip description for a column in a specific dataset."""
        base_tooltips = {
            'BurstID': 'List of photon indices that make up this burst',
            'Number of Photons (fit window)': 'Total photons in the analysis window',
            'r Experimental': 'Experimental anisotropy value',
            'r Scatter': 'Scatter-corrected anisotropy value',
            'Tau (green)': 'Fluorescence lifetime in nanoseconds'
        }
        
        # Dataset-specific tooltips
        if 'Ng-p-all' in column_name:
            return 'Green channel parallel photons'
        elif 'Ng-s-all' in column_name:
            return 'Green channel perpendicular photons'
        elif 'Nr-p-all' in column_name:
            return 'Red channel parallel photons'
        elif 'Nr-s-all' in column_name:
            return 'Red channel perpendicular photons'
        elif 'Ny-p-all' in column_name:
            return 'Yellow channel parallel photons'
        elif 'Ny-s-all' in column_name:
            return 'Yellow channel perpendicular photons'
        
        # Check base tooltips
        for base_name, tooltip in base_tooltips.items():
            if base_name in column_name:
                return tooltip
        
        return f"Data for {column_name} in {dataset.upper()} dataset"