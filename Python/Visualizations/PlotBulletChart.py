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
def PlotBulletChart(dataframe, 
                    value_column, 
                    group_column,
                    target_value_column=None,
                    list_of_limit_columns=None,
                    value_maximum=None,
                    value_minimum=0,
                    # Bullet chart formatting arguments
                    null_background_color='#dbdbdb',
                    background_color_palette="Blues_r",
                    background_color_palette_reverse=False,
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
                    title_for_plot=None,
                    subtitle_for_plot=None,
                    caption_for_plot=None,
                    data_source_for_plot=None,
                    title_y_indent=1.15,
                    subtitle_y_indent=1.1,
                    caption_y_indent=-0.15):
    # If value_maximum is not specified, use the maximum value among the limit columns
    if value_maximum is None:
        value_maximum = dataframe[list_of_limit_columns].max().max()
    
    # Sort the dataframe by the by the grouping column
    dataframe = dataframe.sort_values(by=group_column, ascending=False)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each row in the dataframe
    for index, row in dataframe.iterrows():
        group = row[group_column]
        value = row[value_column]
        if target_value_column != None:
            target = row[target_value_column]
            
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
            colors = sns.color_palette(background_color_palette, len(list_of_limit_columns) + 2)

            # Rearrange the limit columns in descending order according to each of their values
            sorted_row = row[list_of_limit_columns].sort_values(ascending=False)
            
            # Iterate over each limit column
            for i in range(len(list_of_limit_columns)):
                # Plot the limit column value as a bar
                sorted_row_value = sorted_row[i]
                if i == 0:
                    ax.barh(
                    y=group, 
                    width=value_maximum-value_minimum, 
                    color=colors[i], 
                    height=0.8
                )
                ax.barh(
                    y=group, 
                    width=sorted_row_value-value_minimum, 
                    color=colors[i+1], 
                    height=0.8,
                    edgecolor='white',
                    linewidth=2,
                    alpha=background_alpha,
                )
        
        # Plot the target value as a line in the group
        if target_value_column != None:
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
                y=y_position+0.575,
                s=value_label_format.format(value),
                fontname="Arial",
                fontsize=value_label_font_size,
                color=value_label_color,
                verticalalignment="center",
                horizontalalignment="center",
            )
        
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 30)) for label in y_tick_labels]
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


# # Test the function
# # Sample data
# data = {
#     'Group': ['Pittsburgh', 'Denver', 'Tampa'],
#     'Current Performance': [75, 50, 90],
#     'Target': [76, 55, 92],
#     'Level 1': [30, 20, 40],
#     'Level 2': [50, 40, 60],
#     'Level 3': [80, 70, 90]
# }
# data = pd.DataFrame(data)
# # Generate the bullet chart with specified columns
# PlotBulletChart(
#     dataframe=data, 
#     group_column='Group', 
#     value_column='Current Performance', 
#     target_value_column='Target',
#     list_of_limit_columns=['Level 1', 'Level 2', 'Level 3'],
#     value_maximum=100,
#     title_for_plot="Current Performance",
#     subtitle_for_plot="Shown with target and performance levels",
# )
