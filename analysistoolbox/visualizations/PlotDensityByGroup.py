# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotDensityByGroup(dataframe,
                       value_column_name,
                       grouping_column_name,
                       # Plot formatting arguments
                       color_palette='Set2',
                       show_legend=True,
                       figure_size=(8, 6),
                       # Text formatting arguments
                       title_for_plot=None,
                       subtitle_for_plot=None,
                       caption_for_plot=None,
                       data_source_for_plot=None,
                       show_y_axis=False,
                       title_y_indent=1.1,
                       subtitle_y_indent=1.05,
                       caption_y_indent=-0.15,
                       # Plot saving arguments
                       filepath_to_save_plot=None):
    """
    Generates a density plot for a given dataframe, with the option to group by a specific column.

    Args:
        dataframe (pandas.DataFrame): The dataframe to plot.
        value_column_name (str): The name of the column in the dataframe to use for the x-axis values.
        grouping_column_name (str): The name of the column in the dataframe to use for grouping data.
        color_palette (str, optional): The color palette to use for the plot. Defaults to 'Set2'.
        show_legend (bool, optional): Whether to show the legend. Defaults to True.
        figure_size (tuple, optional): The size of the figure. Defaults to (8, 6).
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis. Defaults to False.
        title_y_indent (float, optional): The y-indent for the title. Defaults to 1.1.
        subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.15.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
        
    # Generate density plot using seaborn
    sns.kdeplot(
        data=dataframe,
        x=value_column_name,
        hue=grouping_column_name,
        fill=True,
        common_norm=False,
        alpha=.4,
        palette=color_palette,
        linewidth=1
    )
    
    # Remove top, left, and right spines. Set bottom spine to dark gray.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color("#262626")
    
    # Remove the legend if show_legend is False
    if show_legend == False:
        ax.legend_.remove()
    
    # Remove the y-axis, and adjust the indent of the plot titles
    if show_y_axis == False:
        ax.axes.get_yaxis().set_visible(False)
        x_indent = 0.015
    else:
        x_indent = -0.005
    
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

