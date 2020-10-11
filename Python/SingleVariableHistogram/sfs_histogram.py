# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 07:49:58 2020

@author: oneno
"""

import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns

# Create histogram function
def sfs_histogram(data,
                  quantitative_variable,
                  readable_label_for_quantitative_variable = None,
                  folder_to_save_plot = None,
                  fill_color = "#3269a8",
                  is_discrete_variable = False):
    # Calculate best number of bins and bin width/size
    sd_of_var = data[quantitative_variable].std()
    min_of_var = data[quantitative_variable].min()
    max_of_var = data[quantitative_variable].max()
    n = len(data[quantitative_variable].index)
    ## number of bins
    numerator = (max_of_var - min_of_var) * (n ** (1/3))
    denominator = 3.5 * sd_of_var
    number_of_bins = numerator / denominator
    try:
        number_of_bins = round(number_of_bins, 0)
        number_of_bins = int(number_of_bins)
    except:
        number_of_bins = 1
    if number_of_bins < 1:
        number_of_bins = 1
    ## bin width 
    numerator = 3.5 * sd_of_var
    denominator = n ** (1/3)
    bin_width = numerator / denominator
    bin_width = round(bin_width, 2)
    # Create readable label for quantitative variable if one isn't specified
    if readable_label_for_quantitative_variable == None:
        readable_label_for_quantitative_variable = quantitative_variable
    
    # Create histogram title
    title_for_plot = "Histogram - " + readable_label_for_quantitative_variable
    
    # Set style theme
    sns.set_style("whitegrid")
    
    # Create histogram
    fig = plt.figure()
    if is_discrete_variable:
        bin_width = round(bin_width, 0)
        if bin_width < 1:
            bin_width = 1
        ax = sns.histplot(data = data,
                          x = quantitative_variable,
                          stat = "count",
                          color = fill_color, 
                          bins = number_of_bins,
                          binwidth = bin_width)
    else:
        ax = sns.histplot(data = data,
                          x = quantitative_variable,
                          stat = "count",
                          color = fill_color,
                          bins = number_of_bins)
    ax.set_title(title_for_plot,
                 loc = 'left')
    ax.set_xlabel(readable_label_for_quantitative_variable)
    
    # Save plot to folder if one is specified
    if folder_to_save_plot != None:
        try:
            plot_filepath = str(folder_to_save_plot) + title_for_plot + ".png"
            plot_filepath = os.path.normpath(plot_filepath)
            fig.savefig(plot_filepath)
        except:
            print("The filpath you entered to save your plot is invalid. Plot not saved.")
    
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

