import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

# Create histogram function
def PlotSingleVariableHistogram(dataframe,
                                list_quantitative_variables,
                                folder_to_save_plot=None,
                                fill_color="#3269a8"):
    # Iterate through the list of quantitative variables
    for quantitative_variable in list_quantitative_variables:
        # Calculate best number of bins and bin width/size
        sd_of_var = dataframe[quantitative_variable].std()
        min_of_var = dataframe[quantitative_variable].min()
        max_of_var = dataframe[quantitative_variable].max()
        n = len(dataframe[quantitative_variable].index)
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
        
        # Set style theme
        sns.set_style("whitegrid")
        
        # Create histogram
        fig = plt.figure()
        ax = sns.histplot(data = dataframe,
                          x = quantitative_variable,
                          stat = "count",
                          color = fill_color,
                          bins = number_of_bins)
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
