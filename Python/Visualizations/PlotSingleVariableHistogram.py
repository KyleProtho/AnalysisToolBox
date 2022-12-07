import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Create histogram function
def PlotSingleVariableHistogram(dataframe,
                                list_of_numeric_variables,
                                folder_to_save_plot=None,
                                fill_color="#3269a8"):
    # Set number of column in grid based on length of list of variables
    number_of_columns = 3
    
    # Set number of rows in grid based on length of list of variables
    number_of_rows = ceil(len(list_of_numeric_variables) / number_of_columns)
    
    # Set size of figure
    size_of_figure = (number_of_columns * 3, number_of_rows * 3)
    
    # Create grid
    fig = plt.figure(figsize=size_of_figure)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Iterate through the list of quantitative variables
    for i in range(len(list_of_numeric_variables)):
        # Get variable name
        quantitative_variable = list_of_numeric_variables[i]
        # Create histogram
        ax = fig.add_subplot(number_of_rows, number_of_columns, i+1)
        ax = sns.histplot(data = dataframe,
                          x = quantitative_variable,
                          stat = "count",
                          color = fill_color)
        ax.set_xlabel(quantitative_variable)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_ylabel(None)
        ax.set(yticklabels=[])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
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
    
# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# PlotSingleVariableHistogram(dataframe=iris,
#                             list_of_numeric_variables=iris.columns)
