# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Create histogram function
def PlotSingleVariableBarChart(dataframe,
                               list_of_categorical_variables,
                               fill_color=None,
                               number_of_plot_grid_columns=4):
    
    # Set number of rows in grid based on length of list of variables
    number_of_plot_grid_rows = ceil(len(list_of_numeric_variables) / number_of_plot_grid_columns)
    
    # Set size of figure
    size_of_figure = (number_of_plot_grid_columns * 6, number_of_plot_grid_rows * 6)
    
    # Create grid
    fig = plt.figure(figsize=size_of_figure)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Iterate through the list of categorical variables
    for i in range(len(list_of_categorical_variables)):
        # Get variable name
        categorical_variable = list_of_categorical_variables[i]
        # Generate bar chart
        ax = fig.add_subplot(number_of_plot_grid_rows, number_of_plot_grid_columns, i+1)
        if fill_color == None:
            ax = sns.countplot(data=dataframe,
                               y=categorical_variable,
                               order=dataframe[categorical_variable].value_counts(ascending=False).index,
                               palette="Set1")
        else:
            ax = sns.countplot(data=dataframe,
                               y=categorical_variable,
                               order=dataframe[categorical_variable].value_counts(ascending=False).index,
                               color=fill_color)
        ax.grid(False)
        # String wrap the variable name
        wrapped_variable_name = "\n".join(quantitative_variable[j:j+30] for j in range(0, len(quantitative_variable), 30))
        ax.set_ylabel(wrapped_variable_name)
        ax.set_xlabel(None)
        ax.set(xticklabels=[])
        ax.tick_params(axis='y', which='major', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.suptitle("Distribution",
                     x=0.125,
                     horizontalalignment='left',
                     verticalalignment='top',
                     fontsize=14)
        
        # Add data labels
        abs_values = dataframe[categorical_variable].value_counts(ascending=False)
        rel_values = dataframe[categorical_variable].value_counts(ascending=False, normalize=True).values * 100
        lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
        ax.bar_label(container=ax.containers[0],
                     labels=lbls,
                     padding=5)
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

