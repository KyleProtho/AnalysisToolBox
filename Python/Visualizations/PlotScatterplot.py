# Load packages
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def PlotScatterPlot(dataframe,
                    outcome_variable,
                    list_of_predictor_variables,
                    grouping_variable=None,
                    fitted_line_type=None,
                    fill_color="#3269a8",
                    folder_to_save_plot=None):
    # Iterate through list of predictors 
    for predictor in list_of_predictor_variables:
        # Draw scatterplot
        if fitted_line_type == None:
            if grouping_variable == None:
                ax = sns.scatterplot(
                    data=dataframe, 
                    x=predictor,
                    y=outcome_variable,
                    color=fill_color
                )
            else:
                ax = sns.scatterplot(
                    data=dataframe, 
                    x=predictor,
                    y=outcome_variable,
                    hue=grouping_variable,
                    palette="Set1"
                )
        elif fitted_line_type == 'straight':
            if grouping_variable == None:
                ax = sns.regplot(
                    data=dataframe,
                    x=predictor,
                    y=outcome_variable,
                    color=fill_color,
                    fit_reg=True
                )
            else:
                ax = sns.lmplot(
                    data=dataframe,
                    x=predictor,
                    y=outcome_variable,
                    hue=grouping_variable,
                    palette="Set1",
                    fit_reg=True
                )
        elif fitted_line_type == 'lowess':
            if grouping_variable == None:
                ax = sns.regplot(data=dataframe,
                        x=predictor,
                        y=outcome_variable,
                        color=fill_color,
                        lowess=True)
            else:
                ax = sns.lmplot(
                    data=dataframe,
                    x=predictor,
                    y=outcome_variable,
                    hue=grouping_variable,
                    palette="Set1",
                    lowess=True
                )
        else:
            raise ValueError("Invalied fitted_line_type argument. Please enter None, 'straight', or 'lowess'.")
        if grouping_variable == None or fitted_line_type == None:
            ax.grid(False)
            ax.tick_params(axis='y', which='major', labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.suptitle(
                "Scatterplot",
                x=0.125,
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=14
            )
            plt.title(
                outcome_variable + " and " + predictor,
                loc='left',
                fontsize=10
            )
        else:
            ax.fig.subplots_adjust(top=.9)
            ax.fig.text(
                x=0.125, 
                y=1, 
                s='Scatterplot', 
                fontsize=14,
                ha='left', 
                va='top'
            )
            ax.fig.text(
                x=0.125, 
                y=0.925, 
                s=outcome_variable + " and " + predictor + ", grouped by " + grouping_variable, 
                fontsize=10,
                ha='left', 
                va='bottom'
            )
            
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
