# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotSingleVariableHistogram(dataframe,
                                value_column_name,
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
                                figure_size=(8, 6),
                                # Plot saving arguments
                                filepath_to_save_plot=None):
    """
    Plots a single variable histogram from a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data to be plotted.
        value_column_name (str): The name of the column in the dataframe to be plotted.
        fill_color (str, optional): The fill color for the histogram. Defaults to "#999999".
        fill_transparency (float, optional): The transparency of the fill color. Defaults to 0.6.
        show_mean (bool, optional): Whether to show the mean on the plot. Defaults to True.
        show_median (bool, optional): Whether to show the median on the plot. Defaults to True.
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis. Defaults to False.
        title_y_indent (float, optional): The y-indent for the title. Defaults to 1.1.
        subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.15.
        figure_size (tuple, optional): The size of the figure. Defaults to (8, 6).
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.

    Returns:
        None
    """
    
    # Check that the column exists in the dataframe.
    if value_column_name not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(value_column_name))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create histogram using seaborn
    sns.histplot(
        data=dataframe,
        x=value_column_name,
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
        mean = dataframe[value_column_name].mean()
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
            # fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
    
    # Show the median if requested
    if show_median:
        # Calculate the median
        median = dataframe[value_column_name].median()
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
            # fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
    
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        # fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        # fontname="Arial",
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
    # plt.xticks(fontname='Arial')
    
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
            # fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.transAxes
        )
    
    # If filepath_to_save_plot is provided, save the plot
    if filepath_to_save_plot != None:
        # Ensure that the filepath ends with '.png' or '.jpg'
        if not filepath_to_save_plot.endswith('.png') and not filepath_to_save_plot.endswith('.jpg'):
            raise ValueError("The filepath to save the plot must end with '.png' or '.jpg'.")
        
        # Save plot
        plt.savefig(
            filepath_to_save_plot,
            bbox_inches="tight"
        )
       
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

