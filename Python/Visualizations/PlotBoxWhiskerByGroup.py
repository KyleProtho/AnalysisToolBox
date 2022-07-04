import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

# Create box whisker function
def PlotBoxWhiskerByGroup(dataframe,
                          outcome_variable,
                          group_variable_1,
                          group_variable_2=None,
                          folder_to_save_plot=None,
                          fill_color=None):
    # Create boxplot
    f, ax = plt.subplots(figsize=(6, 4.5))
    if group_variable_2 != None:
        if fill_color != None:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            hue=group_variable_2, 
                            color=fill_color)
        else:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            hue=group_variable_2, 
                            palette="Set1")
    else:
        if fill_color != None:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            color=fill_color)
        else:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            palette="Set1")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.suptitle(outcome_variable,
                    x=0.125,
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=14)
    subtitle_text = 'by ' + group_variable_1
    if group_variable_2 != None:
        subtitle_text = subtitle_text + ' and ' + group_variable_2
    plt.title(subtitle_text,
              loc='left',
              fontsize=10)
    
    # Save plot to folder if one is specified
    if folder_to_save_plot != None:
        try:
            plot_filepath = str(folder_to_save_plot) + "Box Whisker - " + outcome_variable + subtitle_text + ".png"
            plot_filepath = os.path.normpath(plot_filepath)
            fig.savefig(plot_filepath)
        except:
            print("The filpath you entered to save your plot is invalid. Plot not saved.")
    
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()
