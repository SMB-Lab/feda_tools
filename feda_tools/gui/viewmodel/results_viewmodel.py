"""
ResultsViewModel - Handles output formatting and file saving.

This ViewModel orchestrates the saving of analysis results and provides
utilities for formatting and exporting data.
"""

from typing import Dict, Any, Optional
import pathlib
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from feda_tools.core import data as dat


class ResultsViewModel(QObject):
    """ViewModel for orchestrating results handling and file operations."""
    
    # Signals for communicating with Views/Coordinator
    save_started = pyqtSignal()
    save_completed = pyqtSignal(str)  # Output directory path
    save_error = pyqtSignal(str)
    
    # Progress signals
    file_saved = pyqtSignal(str, str)  # file_type, file_path
    
    def __init__(self):
        super().__init__()
        self.current_output_directory = None
        
    def set_output_directory(self, directory: str) -> None:
        """
        Set the output directory for saving results.
        
        Args:
            directory: Path to output directory
        """
        self.current_output_directory = directory
        
        # Create directory if it doesn't exist
        try:
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.save_error.emit(f"Error creating output directory: {str(e)}")
    
    def save_analysis_results(
        self,
        file_path: str,
        bi4_bur_df: pd.DataFrame,
        bg4_df: pd.DataFrame,
        br4_df: pd.DataFrame,
        by4_df: pd.DataFrame,
        output_directory: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save all analysis results to files.
        
        Args:
            file_path: Original data file path (used for naming)
            bi4_bur_df: Burst statistics DataFrame
            bg4_df: Green channel anisotropy DataFrame
            br4_df: Red channel anisotropy DataFrame  
            by4_df: Yellow channel anisotropy DataFrame
            output_directory: Optional output directory override
            
        Returns:
            Dictionary mapping file types to saved file paths
        """
        try:
            self.save_started.emit()
            
            # Use provided directory or default
            output_dir = output_directory or self.current_output_directory
            if not output_dir:
                raise ValueError("No output directory specified")
            
            # Save all files using core data module
            dat.save_results(output_dir, file_path, bi4_bur_df, bg4_df, br4_df, by4_df)
            
            # Generate file paths for confirmation
            base_filename = pathlib.Path(file_path).stem
            saved_files = {
                'bur': pathlib.Path(output_dir) / f"{base_filename}.bur",
                'bg4': pathlib.Path(output_dir) / f"{base_filename}.bg4",
                'br4': pathlib.Path(output_dir) / f"{base_filename}.br4",
                'by4': pathlib.Path(output_dir) / f"{base_filename}.by4"
            }
            
            # Emit individual file save signals
            for file_type, file_path in saved_files.items():
                self.file_saved.emit(file_type, str(file_path))
            
            # Emit completion signal
            self.save_completed.emit(output_dir)
            
            return {k: str(v) for k, v in saved_files.items()}
            
        except Exception as e:
            self.save_error.emit(str(e))
            raise
    
    def save_individual_file(
        self,
        file_path: str,
        dataframe: pd.DataFrame,
        file_extension: str,
        output_directory: Optional[str] = None
    ) -> str:
        """
        Save an individual DataFrame to a file.
        
        Args:
            file_path: Original data file path (used for naming)
            dataframe: DataFrame to save
            file_extension: File extension (bur, bg4, br4, by4)
            output_directory: Optional output directory override
            
        Returns:
            Path to saved file
        """
        try:
            # Use provided directory or default
            output_dir = output_directory or self.current_output_directory
            if not output_dir:
                raise ValueError("No output directory specified")
            
            # Save individual file
            saved_path = dat.save_individual_file(output_dir, file_path, dataframe, file_extension)
            
            # Emit signals
            self.file_saved.emit(file_extension, saved_path)
            
            return saved_path
            
        except Exception as e:
            self.save_error.emit(f"Error saving {file_extension} file: {str(e)}")
            raise
    
    def get_result_summary(
        self,
        bi4_bur_df: pd.DataFrame,
        bg4_df: pd.DataFrame,
        br4_df: pd.DataFrame,
        by4_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a summary of the analysis results.
        
        Args:
            bi4_bur_df: Burst statistics DataFrame
            bg4_df: Green channel anisotropy DataFrame
            br4_df: Red channel anisotropy DataFrame
            by4_df: Yellow channel anisotropy DataFrame
            
        Returns:
            Dictionary containing result summary
        """
        try:
            summary = {
                'total_bursts': len(bi4_bur_df),
                'avg_photons_per_burst': bi4_bur_df['Number of Photons'].mean() if len(bi4_bur_df) > 0 else 0,
                'avg_duration_ms': bi4_bur_df['Duration (ms)'].mean() if len(bi4_bur_df) > 0 else 0,
                'avg_green_photons': bi4_bur_df['Number of Photons (green)'].mean() if len(bi4_bur_df) > 0 else 0,
                'avg_red_photons': bi4_bur_df['Number of Photons (red)'].mean() if len(bi4_bur_df) > 0 else 0,
                'bg4_records': len(bg4_df),
                'br4_records': len(br4_df),
                'by4_records': len(by4_df)
            }
            
            # Calculate additional statistics if data exists
            if len(bi4_bur_df) > 0:
                summary.update({
                    'avg_sg_khz': bi4_bur_df['Sg (prompt) (kHz)'].mean(),
                    'avg_sr_khz': bi4_bur_df['Sr (prompt) (kHz)'].mean(),
                    'avg_sy_khz': bi4_bur_df['Sy (delay) (kHz)'].mean()
                })
            
            if len(bg4_df) > 0:
                summary.update({
                    'avg_r_exp_green': bg4_df['r Experimental (green)'].mean(),
                    'avg_tau_green': bg4_df['Tau (green)'].mean()
                })
            
            return summary
            
        except Exception as e:
            self.save_error.emit(f"Error generating summary: {str(e)}")
            return {}
    
    def export_summary_report(
        self,
        summary: Dict[str, Any],
        file_path: str,
        output_directory: Optional[str] = None
    ) -> str:
        """
        Export a summary report to a text file.
        
        Args:
            summary: Summary dictionary
            file_path: Original data file path (used for naming)
            output_directory: Optional output directory override
            
        Returns:
            Path to saved report file
        """
        try:
            # Use provided directory or default
            output_dir = output_directory or self.current_output_directory
            if not output_dir:
                raise ValueError("No output directory specified")
            
            # Generate report content
            base_filename = pathlib.Path(file_path).stem
            report_path = pathlib.Path(output_dir) / f"{base_filename}_summary.txt"
            
            with open(report_path, 'w') as f:
                f.write(f"Analysis Summary Report\n")
                f.write(f"======================\n")
                f.write(f"Source File: {pathlib.Path(file_path).name}\n\n")
                
                for key, value in summary.items():
                    if isinstance(value, float):
                        f.write(f"{key.replace('_', ' ').title()}: {value:.3f}\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            self.file_saved.emit("summary", str(report_path))
            
            return str(report_path)
            
        except Exception as e:
            self.save_error.emit(f"Error exporting summary report: {str(e)}")
            raise