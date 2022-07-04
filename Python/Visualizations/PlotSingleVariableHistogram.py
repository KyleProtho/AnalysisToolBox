import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Create histogram function
def PlotSingleVariableHistogram(dataframe,
                                list_quantitative_variables,
                                folder_to_save_plot=None,
                                fill_color="#3269a8"):
    # Iterate through the list of quantitative variables
    for quantitative_variable in list_quantitative_variables:
        # Create histogram
        f, ax = plt.subplots(figsize=(6, 4.5))
        ax = sns.histplot(data = dataframe,
                          x = quantitative_variable,
                          stat = "count",
                          color = fill_color)
        ax.set_xlabel(None)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_ylabel(None)
        ax.set(yticklabels=[])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.suptitle("Distribution",
                     x=0.125,
                     horizontalalignment='left',
                     verticalalignment='top',
                     fontsize=14)
        plt.title(quantitative_variable,
                  loc='left',
                  fontsize=10)
        
        # Save plot to folder if one is specified
        if folder_to_save_plot != None:
            try:
                plot_filepath = str(folder_to_save_plot) + "Histogram - " + quantitative_variable + ".png"
                plot_filepath = os.path.normpath(plot_filepath)
                fig.savefig(plot_filepath)
            except:
                print("The filpath you entered to save your plot is invalid. Plot not saved.")
        
        # Show plot
        plt.show()
        
        # Clear plot
        plt.clf()
