from typing import Dict, List, Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy
import scipy.stats

import pathlib
import tttrlib
import os
import tqdm

def update_tttr_dict(
    df: pd.DataFrame,
    data_path: pathlib.Path,
    tttrs: Dict[str, tttrlib.TTTR] = dict(),
    file_type: str = "PTU"
):
    for ff, fl in zip(df['First File'], df['Last File']):
        try:
            tttr = tttrs[ff]
        except KeyError:
            fn = str(data_path / ff)
            tttr = tttrlib.TTTR(fn, file_type)
            tttrs[ff] = tttr    
    return tttrs

def read_analysis(
    paris_path : pathlib.Path,
    paths: List[str] = ['bg4', 'bi4_bur'], #
    file_endings: List[str] = ['bg4', 'bur'],  # 
    file_type: str = "PTU"
) -> (pd.DataFrame, Dict[str, tttrlib.TTTR]):
    
    info_path = paris_path / 'Info'
    data_path = paris_path.parent

    dfs = list()
    for path, ending in zip(paths, file_endings):
        frames = list()
        for fn in sorted((paris_path / path).glob('*.%s' % ending)):
            df = pd.read_csv(fn, sep='\t')
            df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            frames.append(df)
        path_df = pd.concat(frames)
        dfs.append(path_df)
    df = pd.concat(dfs, axis=1)
    
#     df = df.dropna()
    
    # Loop through each column and attempt to convert to numeric
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            # Handle exceptions, e.g., if the column contains non-numeric values
            print(f"Could not convert {column} to numeric")
    tttrs = dict()
    update_tttr_dict(df, data_path, tttrs, file_type)
    return df, tttrs