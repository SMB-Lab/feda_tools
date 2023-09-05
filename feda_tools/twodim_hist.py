"""
Author - Frank Duffy

"""

import argparse
import os
import yaml

calc_list = [
    "Mean Macro Time (sec)",
    "Sg/Sr (prompt)",
    "S(prompt)/S(total)"
]

def make_2dhist(args=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('config_file', type=str)
    parsed_args = parser.parse_args(args)

    print((parsed_args.data_folder, parsed_args.config_file))
    
    with open(parsed_args.config_file) as stream:
        try:
            # Conversts yaml doc to python object
            plot_batch = yaml.safe_load(stream)
        
            #Print the dict
            print(d)
        except yaml.YAMLError as e:
            print(e)
    
    for plot in plot_batch:
        xlabel = plot_batch[plot]['xlabel']
        xfolder = plot_batch[plot]['xfolder']
        ylabel = plot_batch[plot]['ylabel']
        yfolder = plot_batch[plot]['yfolder']
        

 