import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotTwoVariableHeatmap(dataframe,
                           categorical_variable_1,
                           categorical_variable_2,
                           color_palette="Blues",
                           show_legend=False):
    # Create contingency table
    contingency_table = pd.crosstab(
        dataframe[categorical_variable_1],
        dataframe[categorical_variable_2],
        normalize="columns"
    )
    
    # Plot contingency table in a heatmap using seaborn
    sns.heatmap(
        contingency_table,
        cmap=color_palette,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=show_legend
    )
    
    # Set title of plot
    plt.title("Contingency Table for " + categorical_variable_1 + " and " + categorical_variable_2)
    
    # Show plot
    plt.show()
