# Load packages
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotBulletChart(dataframe, 
                    value_column_name, 
                    grouping_column_name,
                    target_value_column_name=None,
                    list_of_limit_columns=None,
                    value_maximum=None,
                    value_minimum=0,
                    # Bullet chart formatting arguments
                    display_order_list=None,
                    null_background_color='#dbdbdb',
                    background_color_palette=None,
                    background_alpha=0.8,
                    value_dot_size=200,
                    value_dot_color="#383838",
                    value_dot_color_by_level=False,
                    value_dot_outline_color="#ffffff",
                    value_dot_outline_width=2,
                    target_line_color='#bf3228',
                    target_line_width=4,
                    figure_size=(8, 6),
                    # Text formatting arguments
                    show_value_labels=True,
                    value_label_color="#262626",
                    value_label_format="{:.0f}",
                    value_label_font_size=12,
                    value_label_spacing=.65,
                    value_label_padding=0.3,
                    value_label_background_color="#ffffff",
                    value_label_background_alpha=0.85,
                    title_for_plot=None,
                    subtitle_for_plot=None,
                    caption_for_plot=None,
                    data_source_for_plot=None,
                    title_y_indent=1.15,
                    subtitle_y_indent=1.1,
                    caption_y_indent=-0.15,
                    # Plot saving arguments
                    filepath_to_save_plot=None):
    """
    Plots a bullet chart using the specified dataframe, value column, and group column.

    Args:
        dataframe: pandas dataframe. The dataframe containing the data to be plotted.
        value_column_name: str. The name of the column in the dataframe containing the values to be plotted.
        grouping_column_name: str. The name of the column in the dataframe containing the groups to be plotted.
        target_value_column_name: str, optional. The name of the column in the dataframe containing the target values to be plotted. Default is None.
        list_of_limit_columns: list of str, optional. A list of the names of the columns in the dataframe containing the limit values to be plotted. Default is None.
        value_maximum: float, optional. The maximum value to be plotted on the chart. If None, the maximum value among the limit columns is used. Default is None.
        value_minimum: float, optional. The minimum value to be plotted on the chart. Default is 0.
        display_order_list: list of str, optional. A list of the groups in the dataframe in the order they should be plotted. Default is None.
        null_background_color: str, optional. The color of the background area of the chart where there is no data. Default is '#dbdbdb'.
        background_color_palette: str, optional. The color palette to be used for the limit columns. If None, a greyscale palette is used. Default is None.
        background_alpha: float, optional. The alpha value of the limit column colors. Default is 0.8.
        value_dot_size: int, optional. The size of the dot representing the value on the chart. Default is 200.
        value_dot_color: str, optional. The color of the dot representing the value on the chart. Default is "#383838".
        value_dot_outline_color: str, optional. The color of the outline of the dot representing the value on the chart. Default is "#ffffff".
        value_dot_outline_width: int, optional. The width of the outline of the dot representing the value on the chart. Default is 2.
        target_line_color: str, optional. The color of the line representing the target value on the chart. Default is '#bf3228'.
        target_line_width: int, optional. The width of the line representing the target value on the chart. Default is 4.
        figure_size: tuple of int, optional. The size of the figure containing the chart. Default is (8, 6).
        show_value_labels: bool, optional. Whether or not to show data labels for the value on the chart. Default is True.
        value_label_color: str, optional. The color of the data labels for the value on the chart. Default is "#262626".
        value_label_format: str, optional. The format string for the data labels for the value on the chart. Default is "{:.0f}".
        value_label_font_size: int, optional. The font size of the data labels for the value on the chart. Default is 12.
        value_label_spacing: float, optional. The spacing between the value and its data label. Default is 0.65.
        title_for_plot: str, optional. The title of the chart. Default is None.
        subtitle_for_plot: str, optional. The subtitle of the chart. Default is None.
        caption_for_plot: str, optional. The caption of the chart. Default is None.
        data_source_for_plot: str, optional. The data source of the chart. Default is None.
        title_y_indent: float, optional. The y-indent of the title of the chart. Default is 1.15.
        subtitle_y_indent: float, optional. The y-indent of the subtitle of the chart. Default is 1.1.
        caption_y_indent: float, optional. The y-indent of the caption of the chart. Default is -0.15.
        filepath_to_save_plot: str, optional. The filepath to save the plot. Default is None.
    """
    
    # Ensure that the number of unique values in the group column is equal to the number of rows in the dataframe
    if len(dataframe[grouping_column_name].unique()) != len(dataframe):
        raise ValueError("The number of unique values in the group column must be equal to the number of rows in the dataframe.")
    
    # If value_maximum is not specified, use the maximum value among the limit columns
    if value_maximum is None:
        if list_of_limit_columns is not None:
            value_maximum = dataframe[list_of_limit_columns].max().max()
        else:
            value_maximum = dataframe[value_column_name].max()
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[grouping_column_name].unique())):
            raise ValueError("display_order_list must contain all of the categories in the " + grouping_column_name + " column of dataframe.")
        else:
            # Reverse the display_order_list
            display_order_list = display_order_list[::-1]
            # Order the dataframe by the display_order_list
            dataframe[grouping_column_name] = pd.Categorical(dataframe[grouping_column_name], categories=display_order_list, ordered=True)
            dataframe = dataframe.sort_values(grouping_column_name)
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=True)[grouping_column_name].unique()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    # Iterate over each row in the dataframe
    for index, row in dataframe.iterrows():
        group = row[grouping_column_name]
        value = row[value_column_name]
        if target_value_column_name != None:
            target = row[target_value_column_name]
            
        # Plot the background area color
        bars = ax.barh(
            y=group, 
            width=value_maximum-value_minimum, 
            color=null_background_color, 
            height=0.8
        )
        
        # Get the y-position of the background area color
        y_position = bars[0].get_y()
        
        # If limit columns are specified, plot the limit columns
        if list_of_limit_columns != None:
            # Get the colors from the specified color palette, using the number of limit columns
            if background_color_palette == None or value_dot_color_by_level == True:
                # Create a greyscale palette
                colors = sns.light_palette("#acacad", len(list_of_limit_columns) + 1)
                # Reverse the palette
                colors = colors[::-1]
            else:
                colors = sns.color_palette(background_color_palette, len(list_of_limit_columns) + 1)

            # Rearrange the limit columns in descending order according to each of their values
            sorted_row = row[list_of_limit_columns].sort_values(ascending=False)
            
            # Iterate over each limit column
            for i in range(len(list_of_limit_columns)):
                # Plot the limit column value as a bar
                sorted_row_value = sorted_row[i]
                if i == 0 and np.isnan(sorted_row_value) == False:
                    ax.barh(
                        y=group, 
                        width=value_maximum-value_minimum, 
                        color=colors[i], 
                        height=0.8
                    )
                # If the limit column value is not null plot the limit column value as a bar:
                if np.isnan(sorted_row_value) == False:
                    ax.barh(
                        y=group, 
                        width=sorted_row_value-value_minimum, 
                        color=colors[i+1], 
                        height=0.8,
                        edgecolor='white',
                        linewidth=2,
                        alpha=background_alpha,
                    )
                else:
                    continue
        
        # Plot the target value as a line in the group
        if target_value_column_name != None:
            ax.plot(
                [target, target], 
                [y_position+0.2, y_position+0.6], 
                color=target_line_color, 
                linewidth=target_line_width
            )

        # Plot value as a large dot
        if value_dot_color_by_level==False:
            ax.scatter(
                value, 
                group, 
                color=value_dot_color, 
                s=value_dot_size,
                edgecolor=value_dot_outline_color,
                linewidth=value_dot_outline_width
            )
        else:
            # Get the colors from the specified color palette, using the number of limit columns
            colors = sns.color_palette(background_color_palette, len(list_of_limit_columns) + 1)
            
            # Iterate through each row in the dataframe
            for index, row in dataframe.iterrows():
                # Get the value of the limit columns
                limit_values = row[list_of_limit_columns]
                
                # Get the colors from the specified color palette, using the number of limit columns
                colors = sns.color_palette(background_color_palette, len(list_of_limit_columns) + 1)
                
                # Assign the color of the dot based on the value of the limit columns
                for i in range(len(limit_values)):
                    if value <= limit_values[i]:
                        value_dot_color = colors[i+1]
                        break
                    else:
                        value_dot_color = "#383838"
                
                # Plot value as a large dot
                ax.scatter(
                    value, 
                    group, 
                    color=value_dot_color, 
                    s=value_dot_size,
                    edgecolor=value_dot_outline_color,
                    linewidth=value_dot_outline_width
                )
        
        # Add data label for the value, if requested
        if show_value_labels:
            ax.text(
                x=value,
                y=y_position+value_label_spacing,
                s=value_label_format.format(value),
                # fontname="Arial",
                fontsize=value_label_font_size,
                color=value_label_color,
                verticalalignment="center",
                horizontalalignment="center",
                # Add a white background to the data label
                bbox=dict(
                    facecolor=value_label_background_color, 
                    alpha=value_label_background_alpha, 
                    edgecolor='none',
                    boxstyle="round,pad="+str(value_label_padding)
                )
            )
        
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Format and wrap y axis tick labels using textwrap
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 30, break_long_words=False)) for label in ax.get_yticklabels()]
    ax.set_yticklabels(
        wrapped_y_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626",
    )
    
    # Set the x indent of the plot titles and captions
    # Get longest y tick label
    longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
    if len(longest_y_tick_label) >= 30:
        x_indent = -0.3
    else:
        x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)
    
    # Change x-axis colors to "#666666"
    ax.tick_params(axis='x', colors="#666666")
    ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
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

