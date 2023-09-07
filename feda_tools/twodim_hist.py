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

    # Conversts yaml doc to python object
    plot_dict = yaml.safe_load(yaml_file)
    return plot_dict

def arg_check(arg):
    path = str(arg)

    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(path + ' could not be found. ' + 
                                         'Check for typos or for errors in your relative path string')

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=arg_check)
    parser.add_argument('plot_file', type=arg_check)
    parsed_args = parser.parse_args(args)
    print((parsed_args.data_folder, parsed_args.plot_file))

    return parsed_args.data_folder, parsed_args.plot_file

def make_2dhist(args=None):
    
    data_folder, plot_file = parse_args(args)
    
    with open(plot_file) as yaml_file:
        plot_dict = get_plot_dict(yaml_file)
    
    for plot in plot_dict:
        xlabel = plot_dict[plot]['xlabel']
        xfolder = plot_dict[plot]['xfolder']
        ylabel = plot_dict[plot]['ylabel']
        yfolder = plot_dict[plot]['yfolder']
        print(xlabel)
        

 