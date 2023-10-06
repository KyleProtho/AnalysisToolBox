# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def PlotSingleVariableHistogram(dataframe,
                                quantitative_variable,
                                # Histogram formatting arguments
                                fill_color="#999999",
                                fill_transparency=0.6,
                                show_mean=True,
                                show_median=True,
                                # Text formatting arguments
                                title_for_plot=None,
                                subtitle_for_plot=None,
                                caption_for_plot=None,
                                data_source_for_plot=None,
                                show_y_axis=False,
                                title_y_indent=1.1,
                                subtitle_y_indent=1.05,
                                caption_y_indent=-0.15,
                                # Plot formatting arguments
                                figure_size=(8, 6)):
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
    
    # Remove top, left, and right spines. Set bottom spine to dark gray.
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#262626")
    
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
    
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        fontname="Arial",
        fontsize=11,
        color="#666666",
        transform=ax.transAxes
    )
    
    # Set x-axis tick label font to Arial, size 9, and color #666666
    ax.tick_params(
        axis='x',
        which='major',
        labelsize=9,
        labelcolor="#666666",
        pad=2,
        bottom=True,
        labelbottom=True
    )
    plt.xticks(fontname='Arial')
    
    # Add a word-wrapped caption if one is provided
    if caption_for_plot != None or data_source_for_plot != None:
        # Create starting point for caption
        wrapped_caption = ""
        
        # Add the caption to the plot, if one is provided
        if caption_for_plot != None:
            # Word wrap the caption without splitting words
            wrapped_caption = textwrap.fill(caption_for_plot, 110, break_long_words=False)
            
        # Add the data source to the caption, if one is provided
        if data_source_for_plot != None:
            wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
        
        # Add the caption to the plot
        ax.text(
            x=x_indent,
            y=caption_y_indent,
            s=wrapped_caption,
            fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.transAxes
        )
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()


# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# # PlotSingleVariableHistogram(
# #     dataframe=iris,
# #     quantitative_variable="sepal length (cm)",
# #     title_for_plot="Sepal Length (cm)",
# #     subtitle_for_plot="Iris Dataset"
# # )
# PlotSingleVariableHistogram(
#     dataframe=iris,
#     quantitative_variable="sepal length (cm)",
#     title_for_plot="Sepal Length (cm)",
#     subtitle_for_plot="Iris Dataset",
#     caption_for_plot="This is a caption that is long enough to wrap onto multiple lines. This is a caption that is long enough to wrap onto multiple lines. This is a caption that is long enough to wrap onto multiple lines.",
#     data_source_for_plot="https://archive.ics.uci.edu/ml/datasets/iris"
# )
