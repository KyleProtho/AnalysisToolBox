# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotClusteredBarChart(dataframe,
                          grouping_column_name_1, 
                          grouping_column_name_2,
                          value_column_name,
                          # Plot formatting arguments
                          color_palette="Set1",
                          fill_transparency=0.8,
                          display_order_list=None,
                          figure_size=(6, 6),
                          show_legend=True,
                          decimal_places_for_data_label=0,
                          # Text formatting arguments
                          title_for_plot=None,
                          subtitle_for_plot=None,
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          x_indent=-0.04,
                          title_y_indent=1.15,
                          subtitle_y_indent=1.1,
                          caption_y_indent=-0.15,
                          # Plot saving arguments
                          filepath_to_save_plot=None):
    """
    Creates a clustered bar chart using seaborn.

    Args:
        dataframe (pd.DataFrame): The data to plot.
        grouping_column_name_1 (str): The name of the first grouping column.
        grouping_column_name_2 (str): The name of the second grouping column.
        value_column_name (str): The name of the value column.
        color_palette (str, optional): The color palette to use for the plot. Defaults to "Set1".
        fill_transparency (float, optional): The transparency of the bars. Defaults to 0.8.
        display_order_list (list, optional): The order to display the 1st categories in the plot. Defaults to None.
        figure_size (tuple, optional): The size of the figure. Defaults to (6, 6).
        show_legend (bool, optional): Whether to show the legend. Defaults to True.
        decimal_places_for_data_label (int, optional): The number of decimal places for the data labels. Defaults to 0.
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        x_indent (float, optional): The x-indent for the title and subtitle. Defaults to -0.04.
        title_y_indent (float, optional): The y-indent for the title. Defaults to 1.15.
        subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.1.
        caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.15.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
    """
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[grouping_column_name_1].unique())):
            raise ValueError("display_order_list must contain all of the categories in the " + grouping_column_name_1 + " column of dataframe.")
        else:
            # Order the dataframe by the display_order_list
            dataframe[grouping_column_name_1] = pd.Categorical(dataframe[grouping_column_name_1], categories=display_order_list, ordered=True)
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=True)[grouping_column_name_1].unique()
    
    # Draw a nested barplot by species and sex
    ax = sns.catplot(
        data=dataframe, 
        kind="bar",
        x=grouping_column_name_1, 
        y=value_column_name, 
        hue=grouping_column_name_2,
        palette=color_palette, 
        alpha=fill_transparency, 
        height=figure_size[1],
        aspect=figure_size[0] / figure_size[1],
        # ci=None,
        errorbar=None,
    )
    
    # Remove the legend if show_legend is False
    if show_legend == False:
        ax.ax.legend_.remove()
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap x axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(grouping_column_name_1, 40, break_long_words=False))  # String wrap the variable name
    ax.ax.set_xlabel(wrapped_variable_name)
    
    # Format and wrap x axis tick labels using textwrap
    x_tick_labels = ax.ax.get_xticklabels()
    wrapped_x_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 20, break_long_words=False)) for label in x_tick_labels]
    ax.ax.set_xticklabels(
        wrapped_x_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Change x-axis colors
    ax.ax.tick_params(axis='x', colors="#262626")
    ax.ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.ax.spines['bottom'].set_visible(False)
    ax.ax.spines['right'].set_visible(False)
    ax.ax.spines['left'].set_visible(False)
    
    # Remove the y-axis tick text
    ax.ax.set_yticklabels([])
    
    # Add data labels to each bar in the plot
    # Get minimum value absolute value of the data
    min_value = dataframe[value_column_name].abs().min()
    for p in ax.ax.patches:
        # Get the height of each bar
        height = p.get_height()
        
        # Format the height to the specified number of decimal places
        height = round(height, decimal_places_for_data_label)
        
        # Add the data label to the plot, using the specified number of decimal places
        value_label = "{:,.{decimal_places}f}".format(height, decimal_places=decimal_places_for_data_label)
        if height != 0:
            ax.ax.text(
                x=p.get_x() + p.get_width() / 2.,
                y=height + (min_value * 0.05),
                s=value_label,
                ha="center",
                # fontname="Arial",
                fontsize=10,
                color="#262626"
            )
        
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        # fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        # fontname="Arial",
        fontsize=11,
        color="#666666",
        transform=ax.ax.transAxes
    )
    
        
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
        ax.ax.text(
            x=x_indent,
            y=caption_y_indent,
            s=wrapped_caption,
            # fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.ax.transAxes
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

