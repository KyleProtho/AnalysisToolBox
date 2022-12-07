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
                                fill_color="#3269a8",
                                number_of_plot_grid_columns=4):
    
    # Set number of rows in grid based on length of list of variables
    number_of_plot_grid_rows = ceil(len(list_of_numeric_variables) / number_of_plot_grid_columns)
    
    # Set size of figure
    size_of_figure = (number_of_plot_grid_columns * 3, number_of_plot_grid_rows * 3)
    
    # Create grid
    fig = plt.figure(figsize=size_of_figure)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Iterate through the list of quantitative variables
    for i in range(len(list_of_numeric_variables)):
        # Get variable name
        quantitative_variable = list_of_numeric_variables[i]
        # Create histogram
        ax = fig.add_subplot(number_of_plot_grid_rows, number_of_plot_grid_columns, i+1)
        ax = sns.histplot(data=dataframe,
                          x=quantitative_variable,
                          stat="count",
                          color=fill_color)
        # String wrap the variable name
        wrapped_variable_name = "\n".join(quantitative_variable[j:j+30] for j in range(0, len(quantitative_variable), 30))
        ax.set_xlabel(wrapped_variable_name)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_ylabel(None)
        ax.set(yticklabels=[])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()
    
# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# PlotSingleVariableHistogram(dataframe=iris,
#                             list_of_numeric_variables=iris.columns)
