"""
Author - Frank Duffy

"""

import argparse
import os
import yaml
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np

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
        df = get_data(data_folder + "\\bi4_bur" )
        df["Mean Macro Time (ms)"] = df["Mean Macro Time (ms)"].div(1000)
        df = df.rename(columns={"Mean Macro Time (ms)": "Mean Macro Time (sec)"})
        return df
    
    elif label == "Sg/Sr (prompt)":
        df = get_data(data_folder + "\\bi4_bur")
        df[label] = df["Green Count Rate (KHz)"].div(df["S prompt red (kHz) | 0-200"])
        # df[label] = np.log(df[label])
        return df

def clean_data(df):
    if "Number of Photons" in df.columns:
        df = df.loc[df["Number of Photons"] > 0]
    elif "Number of Photons (fit window) (green)" in df.columns:
        df = df.loc[df["Number of Photons (fit window) (green)"] > 0]
    elif "Number of Photons (fit window) (red)" in df.columns:
        df = df.loc[df["Number of Photons (fit window) (red)"] > 0]
    elif "Number of Photons (fit window) (yellow)" in df.columns:
        df = df.loc[df["Number of Photons (fit window) (yellow)"] > 0]
    
    if "Unnamed: 14" in df.columns:
        df.drop(labels="Unnamed: 14", axis = 1, inplace=True)
        # df = df[df["Tau (yellow)"].between(1,6)]
        # df = df[df["r Scatter (yellow)"].between(-0.2,1.5)]
    
    # if "TGX_TRR" in df.columns:
    #     df = df[df["TGX_TRR"].between(-5,5)]

    print(df)
    df.replace([np.inf, -np.inf], np.nan, inplace =True)
    print(df)
    df.dropna(inplace = True)
    print(df)

    return df

def make_plot(x, y, xlabel, ylabel, xrange, yrange, bins):

    n_binsx = bins["x"]
    n_binsy = bins["y"] 
    c_map = 'gist_ncar_r'

    # Create a Figure, which doesn't have to be square.
    fig = plt.figure(layout='constrained')
    
    # Create the main axes, leaving 25% of the figure space at the top and on the
    # right to position marginals.
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    
    # The main axes' aspect can be fixed.
    ax.set(aspect="auto")
    ax_histx = ax.inset_axes([0, 1.05, 1.0, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # Draw the scatter plot and marginals.
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if xrange != "auto":
        n_binsx = np.linspace(xrange["min"], xrange["max"], num=n_binsx)

    if yrange != "auto":
        n_binsy = np.linspace(yrange["min"], yrange["max"], num=n_binsy)
 
    if ylabel == "Sg/Sr (prompt)":
        n_binsy = np.geomspace(np.min(y), np.max(y), num=n_binsy)
        plt.yscale("log")
 
    # the 2d hist plot:
    h = ax.hist2d(x, y, bins = [n_binsx, n_binsy], cmap = c_map)
    hist_values_2d = h[0]
    mappable = h[3]
    fig.colorbar(mappable, ax=ax, location='left')


    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=n_binsx)
    ax_histy.hist(y, bins=n_binsy, orientation='horizontal')

    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)

    return fig, ax, hist_values_2d

def make_2dhist(args=None):
    
    data_folder, plot_file = parse_args(args)
    
    with open(plot_file) as yaml_file:
        plot_dict = get_plot_dict(yaml_file)
    
    for plot in plot_dict:
        
        xlabel = plot_dict[plot]['xlabel']
        ylabel = plot_dict[plot]['ylabel']
        xrange = plot_dict[plot]['xrange']
        
        print("Plotting (" + xlabel + ", " + ylabel + ")")

        # check if coordinate is a calculation
        if xlabel in calc_list:
            print("Calculating " + xlabel)
            x_df = get_calc(xlabel, data_folder)
        else:
            xfolder = plot_dict[plot]['xfolder']
            print("Getting " + xlabel + " from " + xfolder)
            xdata_folder = data_folder + "\\" + xfolder
            x_df = get_data(xdata_folder)

        if ylabel in calc_list:
            print("Calculating " + ylabel)
            y_df = get_calc(ylabel, data_folder)
        else:
            yfolder = plot_dict[plot]['yfolder']
            print("Getting " + ylabel + " from " + yfolder)
            ydata_folder = data_folder + "\\" + yfolder
            y_df = get_data(ydata_folder)

        # clean the data i.e. remove photon counts == 0, ignore NaN and inf, etc.
        # x_df = clean_data(x_df)
        # y_df = clean_data(y_df)

        print(x_df[xlabel])
        print(y_df[ylabel])

        print(x_df)

        # dataset = pd.concat([x_df, y_df[ylabel]], axis = 1)
        if np.array_equal(x_df, y_df):
            # same dataset, just take x_df
            print("same data set")
            dataset = x_df
        elif ylabel in x_df.columns:
            # conflict, assume we'd rather have the ylabel column in y_df
            print(ylabel + " in x_df")
            x_df.drop(ylabel, axis = 1, inplace=True)
            dataset = x_df.join(y_df[ylabel])
        else:
            # no conflicts, just join
            print("***No conflicts***")
            dataset = x_df.join(y_df[ylabel])
        
        print(dataset)

        dataset = clean_data(dataset)
        print(dataset)

        make_plot(dataset[xlabel].to_numpy(), dataset[ylabel].to_numpy(), xlabel, xrange, ylabel)
    
    plt.show()

        

 
