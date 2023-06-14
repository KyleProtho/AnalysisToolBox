import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotSingleVariableHistogram(dataframe,
                                quantitative_variable,
                                fill_color="#3269a8",
                                title_for_plot=None,
                                subtitle_for_plot=None,
                                show_mean=True,
                                show_median=True,
                                figure_size=(8, 6),
                                fill_transparency=0.5,
                                show_y_axis=False):
    
    # Check that the column exists in the dataframe.
    if quantitative_variable not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(quantitative_variable))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create histogram using seaborn
    sns.histplot(
        data=dataframe,
        x=quantitative_variable,
        color=fill_color,
        alpha=fill_transparency,
    )
    
    # Remove all spines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove the y-axis, and adjust the indent of the plot titles
    if show_y_axis == False:
        ax.axes.get_yaxis().set_visible(False)
        x_indent = 0.015
    else:
        x_indent = -0.005
    
    # Remove the y-axis label
    ax.set_ylabel(None)
    
    # Remove the x-axis label
    ax.set_xlabel(None)
    
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.text(
        x=x_indent,
        y=1.1,
        s=title_for_plot,
        fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.text(
        x=x_indent,
        y=1.05,
        s=subtitle_for_plot,
        fontname="Arial",
        fontsize=11,
        color="#666666",
        transform=ax.transAxes
    )
    
    # Show the mean if requested
    if show_mean:
        # Calculate the mean
        mean = dataframe[quantitative_variable].mean()
        # Show the mean as a vertical line with a label
        ax.axvline(
            x=mean,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=mean, 
            y=plt.ylim()[1] * 0.97, 
            s='Mean: {:.2f}'.format(mean),
            horizontalalignment='center',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
    
    # Show the median if requested
    if show_median:
        # Calculate the median
        median = dataframe[quantitative_variable].median()
        # Show the median as a vertical line with a label
        ax.axvline(
            x=median,
            ymax=0.90-.02,
            color="#262626",
            linestyle=":",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=median,
            y=plt.ylim()[1] * .90,
            s='Median: {:.2f}'.format(median),
            horizontalalignment='center',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()


# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# PlotSingleVariableHistogram(
#     dataframe=iris,
#     quantitative_variable="sepal length (cm)",
#     title_for_plot="Sepal Length (cm)",
#     subtitle_for_plot="Iris Dataset"
# )
