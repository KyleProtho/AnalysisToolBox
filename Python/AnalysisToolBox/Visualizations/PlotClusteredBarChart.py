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
def PlotClusteredBarChart(dataframe,
                          grouping_column_1, 
                          grouping_column_2,
                          value_column,
                          # Plot formatting arguments
                          color_palette="Set1",
                          fill_transparency=0.8,
                          figure_size=(6, 6),
                          # Text formatting arguments
                          title_for_plot=None,
                          subtitle_for_plot=None,
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          x_indent=-0.04,
                          title_y_indent=1.15,
                          subtitle_y_indent=1.1,
                          caption_y_indent=-0.15,
                          decimal_places_for_data_label=0):
    """
    This function creates a clustered bar chart using seaborn. 
    It takes a pandas DataFrame and column names for grouping and value, along with various optional parameters for plot formatting and text formatting. 

    Args:
    dataframe (pd.DataFrame): The data to plot.
    grouping_column_1 (str): The name of the first grouping column.
    grouping_column_2 (str): The name of the second grouping column.
    value_column (str): The name of the value column.
    color_palette (str, optional): The color palette to use for the plot. Defaults to "Set1".
    fill_transparency (float, optional): The transparency of the bars. Defaults to 0.8.
    figure_size (tuple, optional): The size of the figure. Defaults to (6, 6).
    title_for_plot (str, optional): The title for the plot. Defaults to None.
    subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
    caption_for_plot (str, optional): The caption for the plot. Defaults to None.
    data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
    x_indent (float, optional): The x-indent for the title and subtitle. Defaults to -0.04.
    title_y_indent (float, optional): The y-indent for the title. Defaults to 1.15.
    subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.1.
    caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.15.
    decimal_places_for_data_label (int, optional): The number of decimal places for the data labels. Defaults to 0.

    Returns:
    None. The function creates and displays a plot.
    """
    
    # Draw a nested barplot by species and sex
    ax = sns.catplot(
        data=dataframe, 
        kind="bar",
        x=grouping_column_1, 
        y=value_column, 
        hue=grouping_column_2,
        palette=color_palette, 
        alpha=fill_transparency, 
        height=figure_size[1],
        aspect=figure_size[0] / figure_size[1],
        errorbar=None
    )
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap x axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(grouping_column_1, 30))  # String wrap the variable name
    ax.ax.set_xlabel(wrapped_variable_name)
    
    # Format and wrap x axis tick labels using textwrap
    x_tick_labels = ax.ax.get_xticklabels()
    wrapped_x_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 30)) for label in x_tick_labels]
    ax.ax.set_xticklabels(wrapped_x_tick_labels, fontsize=10, fontname="Arial", color="#262626")
    
    # Change x-axis colors to "#666666"
    ax.ax.tick_params(axis='x', colors="#666666")
    ax.ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.ax.spines['bottom'].set_visible(False)
    ax.ax.spines['right'].set_visible(False)
    ax.ax.spines['left'].set_visible(False)
    
    # Remove the y-axis tick text
    ax.ax.set_yticklabels([])
    
    # Add data labels to each bar in the plot
    # Get minimum value absolute value of the data
    min_value = dataframe[value_column].abs().min()
    for p in ax.ax.patches:
        # Get the height of each bar
        height = p.get_height()
        
        # Format the height to the specified number of decimal places
        height = round(height, decimal_places_for_data_label)
        
        # Add the data label to the plot, using the specified number of decimal places
        value_label = "{:,.{decimal_places}f}".format(height, decimal_places=decimal_places_for_data_label)
        ax.ax.text(
            x=p.get_x() + p.get_width() / 2.,
            y=height + (min_value * 0.05),
            s=value_label,
            ha="center",
            fontname="Arial",
            fontsize=10,
            color="#262626"
        )
        
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        fontname="Arial",
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
            fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.ax.transAxes
        )
    
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()


# # Test the function
# data = {
#     "category": ["A", "A", "B", "B"],
#     "group": ["X", "Y", "X", "Y"],
#     "value": [1, 2, 3, 4],
#     "value_big": [10, 20, 30, 15],
#     "value_small": [0.1, 0.2, 0.3, 0.1]
# }
# data = pd.DataFrame(data)
# # PlotClusteredBarChart(
# #     dataframe=data,
# #     grouping_column_1="category", 
# #     grouping_column_2="group",
# #     value_column="value",
# #     title_for_plot="Clustered Bar Chart",
# #     subtitle_for_plot="Using totally made-up data",
# # )
# # PlotClusteredBarChart(
# #     dataframe=data,
# #     grouping_column_1="category", 
# #     grouping_column_2="group",
# #     value_column="value_big",
# #     title_for_plot="Clustered Bar Chart",
# #     subtitle_for_plot="Using totally made-up data",
# # )
# PlotClusteredBarChart(
#     dataframe=data,
#     grouping_column_1="category", 
#     grouping_column_2="group",
#     value_column="value_small",
#     title_for_plot="Clustered Bar Chart",
#     subtitle_for_plot="Using totally made-up data",
#     decimal_places_for_data_label=1
# )