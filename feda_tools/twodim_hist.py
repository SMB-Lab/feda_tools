"""
Author - Frank Duffy

"""

import argparse
import os
import yaml
import pandas as pd

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

def get_data(data_folder):
    print("Getting data in " + data_folder)
    # print(os.getcwd())
    df_list = []
    for file in os.listdir(data_folder):
        df_list.append(pd.read_csv(data_folder + "\\" + file, sep = '\t'))
    df = pd.concat(df_list)
    # print(df)
    return df

def get_calc(label, data_folder):
    """
    perform the requisite calculation on the data frame and return it
    """
    
    if label == "Mean Macro Time (sec)":
        
        # data_folder = data_folder 
        df = get_data(data_folder)
        df["Mean Macro Time (ms)"] = df["Mean Macro Time (ms)"].div(1000)
        calc_df = df.rename(columns={"Mean Macro Time (ms)": "Mean Macro Time (sec)"})
        
        # get the MMT (secs) column as a df and return
        calc = calc_df[[label]]
        return calc

def make_2dhist(args=None):
    
    data_folder, plot_file = parse_args(args)
    
    with open(plot_file) as yaml_file:
        plot_dict = get_plot_dict(yaml_file)
    
    for plot in plot_dict:
        
        xlabel = plot_dict[plot]['xlabel']
        ylabel = plot_dict[plot]['ylabel']
        
        print("Plotting (" + xlabel + ", " + ylabel + ")")

        # check if coordinate is a calculation
        if xlabel in calc_list:
            print("Calculating " + xlabel)
            x_df = get_calc(xlabel, data_folder)
        else:
            xfolder = plot_dict[plot]['xfolder']
            print("Getting " + xlabel + " from " + xfolder)
            xdata_folder = data_folder + "\\" + xfolder
            x_df = get_data(xdata_folder)[[xlabel]]

        if ylabel in calc_list:
            print("Calculating " + ylabel)
            y_df = get_calc(ylabel, data_folder)
        else:
            yfolder = plot_dict[plot]['yfolder']
            print("Getting " + ylabel + " from " + yfolder)
            ydata_folder = data_folder + "\\" + yfolder
            y_df = get_data(ydata_folder)[[ylabel]]

        dataset = pd.concat([x_df, y_df], axis = 1)
        print(dataset)

        

 