# Load packages
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

# Create histogram function
def PlotSingleVariableBarChart(dataframe,
                               list_categorical_variables,
                               folder_to_save_plot=None,
                               fill_color=None):
    # Iterate through the list of categorical variables
    for categorical_variable in list_categorical_variables:
        # Generate bar chart
        f, ax = plt.subplots(figsize=(6, 4.5))
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
        ax.set_ylabel(None)
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
        plt.title(categorical_variable,
                  loc='left',
                  fontsize=10)
        
        # Add data labels
        abs_values = dataframe[categorical_variable].value_counts(ascending=False)
        rel_values = dataframe[categorical_variable].value_counts(ascending=False, normalize=True).values * 100
        lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
        ax.bar_label(container=ax.containers[0],
                     labels=lbls,
                     padding=5)
        
        # Save plot to folder if one is specified
        if folder_to_save_plot != None:
            try:
                plot_filepath = str(folder_to_save_plot) + "Bar chart - " + categorical_variable + ".png"
                plot_filepath = os.path.normpath(plot_filepath)
                fig.savefig(plot_filepath)
            except:
                print("The filepath you entered to save your plot is invalid. Plot not saved.")
        
        # Show plot
        plt.show()
        
        # Clear plot
        plt.clf()

