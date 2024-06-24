# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotPercentageBarChart(dataframe,
                           categorical_column_name,
                           denominator_column_name,
                           percentage_column_name,
                           # Bar formatting arguments
                           denominator_fill_color="#d4d4d4",
                           percentage_fill_color="#8eb3de",
                           color_palette=None,
                           fill_transparency=0.8,
                           # Plot formatting arguments
                           display_order_list=None,
                           figure_size=(8, 6),
                           # Text formatting arguments
                           title_for_plot=None,
                           subtitle_for_plot=None,
                           caption_for_plot=None,
                           data_source_for_plot=None,
                           title_y_indent=1.15,
                           subtitle_y_indent=1.1,
                           caption_y_indent=-0.15,
                           decimal_places_for_data_label=1,
                           data_label_fontsize=11,
                           data_label_padding=None,
                           # Plot saving arguments
                           filepath_to_save_plot=None):
    
    # Check that the column exists in the dataframe.
    if categorical_column_name not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(categorical_column_name))
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[categorical_column_name].unique())):
            raise ValueError("display_order_list must contain all of the categories in the dataframe.")
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(denominator_column_name, ascending=False)[categorical_column_name]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Calculate the numerator values
    dataframe['Numerator'] = dataframe[percentage_column_name] * dataframe[denominator_column_name]
    
    # Create bar plot for the denominator values
    sns.barplot(
        x=categorical_column_name,
            y=denominator_column_name,
            data=dataframe,
            color=denominator_fill_color,
            order=display_order_list,
            ax=ax
    )
    
    # Create bar plot for the percentage values on top of the denominator values
    if color_palette != None:
        ax = sns.barplot(
            data=dataframe,
            y='Numerator',
            x=categorical_column_name,
            hue=dataframe[categorical_column_name],
            palette=color_palette,
            order=display_order_list,
            alpha=fill_transparency
        )
    else:
        ax = sns.barplot(
            data=dataframe,
            y='Numerator',
            x=categorical_column_name,
            color=percentage_fill_color,
            order=display_order_list,
            alpha=fill_transparency
        )
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Set the y axis ticks
    ax.set_yticks(range(len(dataframe[categorical_column_name].value_counts())))
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in y_tick_labels]
    ax.set_yticklabels(
        wrapped_y_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Change x-axis colors to "#666666"
    ax.tick_params(axis='x', colors="#666666")
    ax.spines['bottom'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Get the maximum value of the y-axis
    if data_label_padding == None:
        max_y = dataframe[categorical_column_name].value_counts().max()
        data_label_padding = max_y * 0.10
    
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


# Test the function
# Create a sample dataframe
data = {
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Denominator': [100, 200, 300, 400, 500],
    'Percentage': [0.1, 0.2, 0.3, 0.4, 0.5]
}
df = pd.DataFrame(data)

# Plot the percentage bar chart
PlotPercentageBarChart(
    dataframe=df,
    categorical_column_name='Category',
    denominator_column_name='Denominator',
    percentage_column_name='Percentage',
    color_palette=None,
    fill_transparency=0.8,
    # Plot formatting arguments
    display_order_list=None,
    figure_size=(8, 6),
    # Text formatting arguments
    title_for_plot="Percentage Bar Chart",
    subtitle_for_plot="This is a sample percentage bar chart.",
    caption_for_plot="This is a sample caption for the plot.",
    data_source_for_plot="This is a sample data source.",
)
    