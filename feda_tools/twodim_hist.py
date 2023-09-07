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
    """
    checks if the arguments provided by the user correspond to paths that exist
    """
    path = str(arg)

    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(path + ' could not be found. ' + 
                                         'Check for typos or for errors in your relative path string')

def parse_args(args):
    """
    parse the arguments provided by the user and return them to the main program
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=arg_check)
    parser.add_argument('plot_file', type=arg_check)
    parsed_args = parser.parse_args(args)
    print((parsed_args.data_folder, parsed_args.plot_file))

    return parsed_args.data_folder, parsed_args.plot_file

def get_calc(label, data_folder):
    """
    perform a calculation on the data frame and return it
    """
    
    if label == "Mean Macro Time (sec)":
        data_folder = "bi4_bur"

        # bur_df["Mean Macro Time (ms)"] = bur_df["Mean Macro Time (ms)"].div(1000)
        # calc_df = bur_df.rename(columns={"Mean Macro Time (ms)": "Mean Macro Time (sec)"})
        # print(calc_df)
        # calc = calc_df[label]
        return None

def get_data(data_folder):
    return None

def make_2dhist(args=None):
    
    data_folder, plot_file = parse_args(args)
    
    with open(plot_file) as yaml_file:
        plot_dict = get_plot_dict(yaml_file)
    
    for plot in plot_dict:
        xlabel = plot_dict[plot]['xlabel']
        xfolder = plot_dict[plot]['xfolder']
        ylabel = plot_dict[plot]['ylabel']
        yfolder = plot_dict[plot]['yfolder']
        

        # check if coordinate is a calculation
        if xlabel in calc_list:
            print("Calculating " + xlabel)
            x_series = get_calc(xlabel, data_folder)
        

 