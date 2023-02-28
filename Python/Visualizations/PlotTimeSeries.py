import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotTimeSeries(dataframe,
                   numeric_variable,
                   time_variable,
                   group_variable=None,
                   line_color="#3269a8",
                   color_palette="Set2",
                   plot_title=None):
    
    # Use Seaborn to create a line plot
    if group_variable is None:
        ax = sns.lineplot(
            data=dataframe,
            x=time_variable,
            y=numeric_variable,
            color=line_color,
            alpha=0.8
        )
    else:
        ax = sns.lineplot(
            data=dataframe,
            x=time_variable,
            y=numeric_variable,
            hue=group_variable,
            palette=color_palette,
            alpha=0.8
        )
    
    # Add plot title
    if plot_title is None:
        plot_title = numeric_variable + " over time"
    ax.set_title(plot_title, fontsize=12, fontweight="bold", fontname="Arial")
    
    # Remove top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Show plot
    plt.show()
    

# # Test the function
# time_series = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
# PlotTimeSeries(
#     dataframe=time_series,
#     numeric_variable='Consumption',
#     time_variable='Date'
# )
