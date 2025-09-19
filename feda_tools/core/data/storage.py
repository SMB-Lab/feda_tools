import os
import pandas as pd
from typing import Optional


def save_results(output_directory, file_path, bi4_bur_df, bg4_df, br4_df=None, by4_df=None):
    """
    Save burst analysis results to tab-separated files.
    
    Args:
        output_directory: Directory where files should be saved
        file_path: Original data file path (used for naming output files)
        bi4_bur_df: DataFrame containing burst statistics (.bur file)
        bg4_df: DataFrame containing green channel anisotropy data (.bg4 file)
        br4_df: DataFrame containing red channel anisotropy data (.br4 file)
        by4_df: DataFrame containing yellow channel anisotropy data (.by4 file)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get base filename for all output files
    base_filename = os.path.splitext(os.path.basename(str(file_path)))[0]
    
    # Save .bur file
    bur_filepath = os.path.join(output_directory, base_filename) + ".bur"
    bi4_bur_df.to_csv(bur_filepath, sep='\t', index=False, float_format='%.6f')
    print(f"Saved {len(bi4_bur_df)} burst records to {bur_filepath}")
    
    # Save .bg4 file
    bg4_filepath = os.path.join(output_directory, base_filename) + ".bg4"
    bg4_df.to_csv(bg4_filepath, sep='\t', index=False, float_format='%.6f')
    print(f"Saved {len(bg4_df)} BG4 records to {bg4_filepath}")
    
    # Save .br4 file if provided
    if br4_df is not None:
        br4_filepath = os.path.join(output_directory, base_filename) + ".br4"
        br4_df.to_csv(br4_filepath, sep='\t', index=False, float_format='%.6f')
        print(f"Saved {len(br4_df)} BR4 records to {br4_filepath}")
    
    # Save .by4 file if provided
    if by4_df is not None:
        by4_filepath = os.path.join(output_directory, base_filename) + ".by4"
        by4_df.to_csv(by4_filepath, sep='\t', index=False, float_format='%.6f')
        print(f"Saved {len(by4_df)} BY4 records to {by4_filepath}")


def save_individual_file(output_directory: str, file_path: str, dataframe: pd.DataFrame, 
                        file_extension: str) -> str:
    """
    Save a single DataFrame to a tab-separated file.
    
    Args:
        output_directory: Directory where file should be saved
        file_path: Original data file path (used for naming output file)
        dataframe: DataFrame to save
        file_extension: File extension (e.g., 'bur', 'bg4', 'br4', 'by4')
        
    Returns:
        Full path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate output file path
    base_filename = os.path.splitext(os.path.basename(str(file_path)))[0]
    output_filepath = os.path.join(output_directory, f"{base_filename}.{file_extension}")
    
    # Save the DataFrame
    dataframe.to_csv(output_filepath, sep='\t', index=False, float_format='%.6f')
    
    return output_filepath