# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:10:50 2020

@author: Kyle Protho
"""

# Load packages
import pandas as pd
import matplotlib as plt
import seaborn as sns
from qbstyles import mpl_style
from os import path

def sfs_histogram(data,
                  quantitative_variable,
                  readable_label_for_quantitative_variable = None,
                  folder_to_save_plot = None,
                  fill_color = "#3269a8"):
    # Calculate best number of bins and bin width/size
    sd_of_var = data[quantitative_variable].std()
    min_of_var = data[quantitative_variable].min()
    max_of_var = data[quantitative_variable].max()
    n = data[quantitative_variable].count()
    ## number of bins
    numerator = (max_of_var - min_of_var) * (n ** (1/3))
    denominator = 3.5 * sd_of_var
    number_of_bins = numerator / denominator
    number_of_bins = round(number_of_bins, 0)
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
    p = sns.histplot(data = data,
                     x = quantitative_variable,
                     stat = "count",
                     color = fill_color,
                     bins = number_of_bins,
                     binwidth = bin_width)
    p.set(title = title_for_plot,
          xlabel = str(readable_label_for_quantitative_variable))
    
    # Save plot to folder if one is specified
    if folder_to_save_plot != None:
        try:
            plot_filepath = str(folder_to_save_plot) + title_for_plot + ".png"
            plot_filepath = path.normpath(plot_filepath)
            p.savefig(plot_filepath)
        except:
            print("The filpath you entered to save your plot is invalid. Plot not saved.")


# Import test data (mtcars)
df_mtcars = pd.read_csv('https://gist.githubusercontent.com/ZeccaLehn/4e06d2575eb9589dbe8c365d61cb056c/raw/64f1660f38ef523b2a1a13be77b002b98665cdfe/mtcars.csv')
df_mtcars.rename(columns = {'Unnamed: 0':'brand'}, 
                 inplace=True)
sfs_histogram(data = df_mtcars,
              quantitative_variable = "mpg")