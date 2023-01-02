# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
from textwrap import wrap
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotSingleVariableBarChart(dataframe,
                               list_of_categorical_variables,
                               fill_color=None,
                               number_of_plot_grid_columns=2,
                               add_rare_category_line=False,
                               rare_category_line_color='#b5b3b3',
                               rare_category_threshold=0.05):
    
    # Set number of rows in grid based on length of list of variables
    number_of_plot_grid_rows = ceil(len(list_of_categorical_variables) / number_of_plot_grid_columns)
    
    # Set size of figure
    size_of_figure = (number_of_plot_grid_columns * 6, number_of_plot_grid_rows * 4)
    
    # Ensure that rare category threshold is between 0 and 1
    if rare_category_threshold < 0 or rare_category_threshold > 1:
        raise ValueError("Rare category threshold must be between 0 and 1.")
    
    # Create grid
    fig = plt.figure(figsize=size_of_figure)
    fig.subplots_adjust(hspace=0.2, wspace=1)
    
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
        # Wrap y axis label
        wrapped_variable_name = "\n".join(categorical_variable[j:j+30] for j in range(0, len(categorical_variable), 30))  # String wrap the variable name
        ax.set_ylabel(wrapped_variable_name)
        
        # Format and wrap y axis tick labels
        y_tick_labels = ax.get_yticklabels()
        wrapped_y_tick_labels = ['\n'.join(wrap(label.get_text(), 30)) for label in y_tick_labels]
        ax.set_yticklabels(wrapped_y_tick_labels)
        ax.tick_params(axis='y', which='major', labelsize=8)
        
        # Remove x axis label
        ax.set_xlabel(None)
        ax.set(xticklabels=[])
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add data labels
        abs_values = dataframe[categorical_variable].value_counts(ascending=False)
        rel_values = dataframe[categorical_variable].value_counts(ascending=False, normalize=True).values * 100
        lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
        ax.bar_label(container=ax.containers[0],
                     labels=lbls,
                     padding=5)
        
        # Add rare category threshold line
        if add_rare_category_line:
            ax.axvline(
                x=rare_category_threshold * dataframe.shape[0],
                color=rare_category_line_color,
                alpha=0.5,
                linestyle='--',
                label='Rare category threshold'
            )
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

# # Test the function
# import numpy as np
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# iris['species'] = iris['species'].astype('category')
# iris['species long label'] = np.where(
#     iris['species'] == 0,
#     "This is a super long label for species 0 just to test the function",
#     iris['species']
# )
# PlotSingleVariableBarChart(dataframe=iris,
#                            list_of_categorical_variables=['species long label'],
#                            number_of_plot_grid_columns=1,
#                            add_rare_category_line=True)
