from typing import Dict, List, Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import tqdm

### define function for extracting the unmasked segments from the thresholded data.
def extract_unmasked_indices(masked_array):
    unmasked_indices_lists = []
    current_indices = []

    # iterate through masked array and collect unmasked index segments
    for i, value in enumerate(masked_array):
        if np.ma.is_masked(value):
            if current_indices:
                unmasked_indices_lists.append(current_indices)
                current_indices = []
        else:
            current_indices.append(i)

    # handle the last segment
    if current_indices:
        unmasked_indices_lists.append(current_indices)

    return unmasked_indices_lists

def calc_running_average(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def calc_interphoton_arrival_times(tttr_data):
    
    # - Each detected photon has a time of detection encoded by the macro time + the micro time. **all_macro_times** and **all_micro_times** are arrays whose index is represents the detected photons in order of detection, while the value represents the associated macro or micro time for each photon.
    # - **macro_res** and **micro_res** represent the resolution of the macro and micro times in seconds.
    # - The **macro time** indicates the time in units of **macro_res** that the excitation laser was last fired directly before this photon was detected.
    # - The **micro time** indicates the amount of time in units of **micro_res** that has elapsed since the excitation laser was last fired at which the photon was detected, i.e. it's the amount of time elapsed from the macro time at which the photon was detected.
    # - The interphoton arrival time is calculated by iterating through **all_macro_times** and **all_micro_times** and calculating the time elapsed between each photon detection event.

    all_macro_times = tttr_data.macro_times
    all_micro_times = tttr_data.micro_times
    macro_res =tttr_data.get_header().macro_time_resolution
    micro_res = tttr_data.get_header().micro_time_resolution
    
    #iterate through macro and micro times to calculate delta time between photon events
    arr_size = len(all_macro_times) - 1
    interphoton_arrival_times = np.zeros(arr_size, dtype = np.float64)
    lw = 0.25
    for i in range(0, len(interphoton_arrival_times)):
        photon_1 = (all_macro_times[i]*macro_res) + (all_micro_times[i]*micro_res)
        photon_2 = (all_macro_times[i+1]*macro_res) + (all_micro_times[i+1]*micro_res)
        interphoton_arrival_times[i] = (photon_2 - photon_1)*1000
        
    return interphoton_arrival_times

def filter_burstids(df, bid_path):
    
    # get bid_df to use as a filter on the passed df.
    bid_df = get_bid_df(bid_path)
    
    # Filter df using bid_df.
    result_df = df.merge(bid_df, on=['First Photon', 'Last Photon', 'First File'], how='inner')
    
    return result_df

# read all of the burst id files in the selected directory and create a dataframe storing 
# their First Photon, Last Photon, and First File values row-wise.

def get_bid_df(bid_path):

    # create dfs list to append each bst file dataframe when reading the directory
    dfs = []
    
    # iterate through the bst files in the specified directory
    for file in sorted((bid_path).glob('*.%s' % 'bst')):
        
        # read each file and create a dataframe
        file_df = pd.read_csv(file, sep ='\t', header=None)
        
        # rename the unnammed columns
        file_df.rename(columns={0: "First Photon", 1: "Last Photon"}, inplace=True)
        
        # get the filename so we can assoc a file to the First File column
        filename = os.path.basename(file).split('.')[0]
        
        # process the filename so that it matches the format in the bur file
        filename = filename.replace("_0", "")
        filename = filename + ".ptu"

        # assign this files First and Last Photon data to the associated file.
        file_df['First File'] = filename
        
        # append the burst file dataframe to a list, will concatenate them all after loop.
        dfs.append(file_df)
        
    # concatenate all the dfs into one 
    flphotons = pd.concat(dfs, axis = 0, ignore_index = True)

    return flphotons