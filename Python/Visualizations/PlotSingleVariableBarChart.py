# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:48:33 2020

@author: oneno
"""

# Load packages
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

# Create histogram function
def PlotSingleVariableBarChart(data,
                               categorical_variable,
                               readable_label_for_categorical_variable = None,
                               folder_to_save_plot = None,
                               fill_color = None):
    # Create readable label for qualitative variable if one isn't specified
    if readable_label_for_categorical_variable == None:
        readable_label_for_categorical_variable = categorical_variable
    
    # Create histogram title
    title_for_plot = "Bar Chart - " + readable_label_for_categorical_variable
    
    # Set style theme
    sns.set_style("whitegrid")
    
    # Generate bar chart
    fig = plt.figure()
    if fill_color == None:
        ax = sns.countplot(data = data,
                           x = categorical_variable, 
                           palette = "Spectral")
    else:
        ax = sns.countplot(data = data,
                           x = categorical_variable,
                           color = fill_color)
    ax.set_title(title_for_plot,
                 loc = 'left')
    ax.set_xlabel(readable_label_for_categorical_variable)
    
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

