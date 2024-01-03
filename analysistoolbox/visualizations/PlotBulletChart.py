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
                    title_for_plot=None,
                    subtitle_for_plot=None,
                    caption_for_plot=None,
                    data_source_for_plot=None,
                    title_y_indent=1.15,
                    subtitle_y_indent=1.1,
                    caption_y_indent=-0.15):
    """
    This function plots a bullet chart using the specified dataframe, value column, and group column.
    The bullet chart is a variation of a bar chart that is used to display a single value in relation to a target value and a set of limit values.
    The function allows for customization of the bullet chart formatting and text formatting.
    
    Args:
        dataframe: pandas dataframe. The dataframe containing the data to be plotted.
        value_column_name: str. The name of the column in the dataframe containing the values to be plotted.
        grouping_column_name: str. The name of the column in the dataframe containing the groups to be plotted.
        target_value_column_name: str, default None. The name of the column in the dataframe containing the target values to be plotted.
        list_of_limit_columns: list of str, default None. A list of the names of the columns in the dataframe containing the limit values to be plotted.
        value_maximum: float, default None. The maximum value to be plotted on the chart. If None, the maximum value among the limit columns is used.
        value_minimum: float, default 0. The minimum value to be plotted on the chart.
        display_order_list: list of str, default None. A list of the groups in the dataframe in the order they should be plotted.
        null_background_color: str, default '#dbdbdb'. The color of the background area of the chart where there is no data.
        background_color_palette: str, default None. The color palette to be used for the limit columns. If None, a greyscale palette is used.
        background_alpha: float, default 0.8. The alpha value of the limit column colors.
        value_dot_size: int, default 200. The size of the dot representing the value on the chart.
        value_dot_color: str, default "#383838". The color of the dot representing the value on the chart.
        value_dot_outline_color: str, default "#ffffff". The color of the outline of the dot representing the value on the chart.
        value_dot_outline_width: int, default 2. The width of the outline of the dot representing the value on the chart.
        target_line_color: str, default '#bf3228'. The color of the line representing the target value on the chart.
        target_line_width: int, default 4. The width of the line representing the target value on the chart.
        figure_size: tuple of int, default (8, 6). The size of the figure containing the chart.
        show_value_labels: bool, default True. Whether or not to show data labels for the value on the chart.
        value_label_color: str, default "#262626". The color of the data labels for the value on the chart.
        value_label_format: str, default "{:.0f}". The format string for the data labels for the value on the chart.
        value_label_font_size: int, default 12. The font size of the data labels for the value on the chart.
        title_for_plot: str, default None. The title of the chart.
        subtitle_for_plot: str, default None. The subtitle of the chart.
        caption_for_plot: str, default None. The caption of the chart.
        data_source_for_plot: str, default None. The data source of the chart.
        title_y_indent: float, default 1.15. The y-indent of the title of the chart.
        subtitle_y_indent: float, default 1.1. The y-indent of the subtitle of the chart.
        caption_y_indent: float, default -0.15. The y-indent of the caption of the chart.
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
            if background_color_palette == None:
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
            )
        
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in y_tick_labels]
    ax.set_yticklabels(wrapped_y_tick_labels, fontsize=10, fontname="Arial", color="#262626")
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
    # Change x-axis colors to "#666666"
    ax.tick_params(axis='x', colors="#666666")
    ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
        
    # Set the x indent of the plot titles and captions
    # Get longest y tick label
    longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
    if len(longest_y_tick_label) >= 30:
        x_indent = -0.3
    else:
        x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)
        
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
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

