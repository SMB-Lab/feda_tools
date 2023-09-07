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

def get_plot_dict(yaml_file):
    """
    gets the plots specified by the user in the provided yaml file.
    """
    try:
        # Conversts yaml doc to python object
        plot_dict = yaml.safe_load(yaml_file)
    except yaml.YAMLError as e:
        print(e)
    return plot_dict

def make_2dhist(args=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('plot_file', type=str)
    parsed_args = parser.parse_args(args)
    print((parsed_args.data_folder, parsed_args.plot_file))
    
    with open(parsed_args.plot_file) as yaml_file:
        plot_dict = get_plot_dict(yaml_file)
    
    for plot in plot_dict:
        xlabel = plot_dict[plot]['xlabel']
        xfolder = plot_dict[plot]['xfolder']
        ylabel = plot_dict[plot]['ylabel']
        yfolder = plot_dict[plot]['yfolder']
        

 