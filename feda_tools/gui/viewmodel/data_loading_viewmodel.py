"""
DataLoadingViewModel - Handles PTU file loading and validation.

This ViewModel orchestrates the loading of PTU files and provides 
validation and error handling for the data loading process.
"""

from typing import Tuple, Optional
import pathlib
from PyQt6.QtCore import QObject, pyqtSignal

from feda_tools.core import data as dat


class DataLoadingViewModel(QObject):
    """ViewModel for orchestrating data loading operations."""
    
    # Signals for communicating with Views/Coordinator
    loading_started = pyqtSignal()
    loading_completed = pyqtSignal()
    loading_error = pyqtSignal(str)
    files_validated = pyqtSignal(bool)  # True if all files valid
    data_loaded = pyqtSignal(object, object, object)  # data_ptu, data_irf, data_bkg
    
    def __init__(self):
        super().__init__()
        self.current_file_paths = {}
        
    def set_file_paths(
        self, 
        file_ptu: str, 
        file_irf: Optional[str] = None, 
        file_bkg: Optional[str] = None
    ) -> None:
        """
        Set the file paths for PTU data loading.
        
        Args:
            file_ptu: Path to main PTU file
            file_irf: Path to IRF file (optional)
            file_bkg: Path to background file (optional)
        """
        self.current_file_paths = {
            'ptu': file_ptu,
            'irf': file_irf,
            'bkg': file_bkg
        }
        
        # Validate file paths
        is_valid = self._validate_file_paths()
        self.files_validated.emit(is_valid)
    
    def load_ptu_data(self) -> Tuple[object, object, object]:
        """
        Load PTU data files using the configured paths.
        
        Returns:
            Tuple of (data_ptu, data_irf, data_bkg)
        """
        try:
            self.loading_started.emit()
            
            if not self._validate_file_paths():
                raise ValueError("Invalid file paths. Please check file locations.")
            
            # Load the main PTU file
            file_ptu = self.current_file_paths['ptu']
            data_ptu = dat.load_ptu_files(file_ptu, None, None)[0]
            
            # Load IRF file if provided
            data_irf = None
            if self.current_file_paths.get('irf'):
                try:
                    _, data_irf, _ = dat.load_ptu_files(
                        file_ptu, 
                        self.current_file_paths['irf'], 
                        None
                    )
                except Exception as e:
                    # IRF loading failed, continue without it
                    print(f"Warning: Could not load IRF file: {e}")
            
            # Load background file if provided
            data_bkg = None
            if self.current_file_paths.get('bkg'):
                try:
                    _, _, data_bkg = dat.load_ptu_files(
                        file_ptu, 
                        None, 
                        self.current_file_paths['bkg']
                    )
                except Exception as e:
                    # Background loading failed, continue without it
                    print(f"Warning: Could not load background file: {e}")
            
            # Emit success signals
            self.data_loaded.emit(data_ptu, data_irf, data_bkg)
            self.loading_completed.emit()
            
            return data_ptu, data_irf, data_bkg
            
        except Exception as e:
            self.loading_error.emit(str(e))
            raise
    
    def get_file_info(self) -> dict:
        """
        Get information about the loaded PTU files.
        
        Returns:
            Dictionary containing file information
        """
        if not self.current_file_paths.get('ptu'):
            return {}
        
        try:
            # Load just the main PTU file to get info
            data_ptu = dat.load_ptu_files(self.current_file_paths['ptu'], None, None)[0]
            
            header = data_ptu.get_header()
            
            return {
                'filename': pathlib.Path(self.current_file_paths['ptu']).name,
                'total_events': len(data_ptu.macro_times),
                'macro_time_resolution': header.macro_time_resolution,
                'micro_time_resolution': header.micro_time_resolution,
                'macro_time_resolution_ns': header.macro_time_resolution * 1e9,
                'micro_time_resolution_ps': header.micro_time_resolution * 1e12,
                'has_irf': self.current_file_paths.get('irf') is not None,
                'has_background': self.current_file_paths.get('bkg') is not None
            }
            
        except Exception as e:
            self.loading_error.emit(f"Error getting file info: {str(e)}")
            return {}
    
    def _validate_file_paths(self) -> bool:
        """
        Validate that the specified file paths exist and are accessible.
        
        Returns:
            True if main PTU file exists, False otherwise
        """
        # Main PTU file is required
        if not self.current_file_paths.get('ptu'):
            return False
            
        ptu_path = pathlib.Path(self.current_file_paths['ptu'])
        if not ptu_path.exists() or not ptu_path.is_file():
            return False
        
        # IRF and background files are optional, but if specified must exist
        for file_type in ['irf', 'bkg']:
            file_path = self.current_file_paths.get(file_type)
            if file_path:
                path = pathlib.Path(file_path)
                if not path.exists() or not path.is_file():
                    print(f"Warning: {file_type.upper()} file does not exist: {file_path}")
                    # Remove invalid path
                    self.current_file_paths[file_type] = None
        
        return True