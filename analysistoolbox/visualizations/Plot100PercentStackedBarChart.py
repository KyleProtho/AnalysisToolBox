# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def Plot100PercentStackedBarChart(dataframe,
                                  group_column_name='Group',
                                  value_column_name='Current Value',
                                  target_value_column_name=None,
                                  # Plot formatting arguments
                                  background_color='#e8e8ed',
                                  color_palette="Set1",
                                  fill_color=None,
                                  fill_transparency=0.8,
                                  figure_size=(8, 6),
                                  # Text formatting arguments
                                  data_label_fontsize=10,
                                  title_for_plot=None,
                                  subtitle_for_plot=None,
                                  caption_for_plot=None,
                                  data_source_for_plot=None,
                                  title_y_indent=1.15,
                                  subtitle_y_indent=1.1,
                                  caption_y_indent=-0.15):
    # Ensure that the number of unique values in the 'Group' column is equal to the number of rows in the DataFrame
    if len(dataframe[group_column_name].unique()) != len(dataframe):
        raise ValueError("The number of unique values in the 'Group' column must be equal to the number of rows in the DataFrame.")
    
    # Calculate the percentage of the Current Value out of the Target Value
    if target_value_column_name != None:
        dataframe['Percentage'] = dataframe[value_column_name] / dataframe[target_value_column_name] * 100
        dataframe['Remaining'] = 100 - dataframe['Percentage']
    else:
        dataframe['Percentage'] = dataframe[value_column_name] / dataframe[value_column_name].sum() * 100
        dataframe['Remaining'] = 100 - dataframe['Percentage']
    
    # Sort the DataFrame based on the 'Group' column to match the user's example
    dataframe.sort_values(by='Group', inplace=True)
    
    # Short by the 'Percentage' column to make the plot look better
    dataframe.sort_values(by='Percentage', inplace=True)
    
    # Set the figure size for better clarity
    plt.figure(figsize=figure_size)
    
    # If fill_color is not provided, use the color_palette
    if fill_color == None:
        # Create a color palette
        color_palette = sns.color_palette(color_palette, len(dataframe))
        
        # Create a list of colors
        colors = [color_palette[i] for i in range(len(dataframe))]
    else:
        # Create a list of colors
        colors = [fill_color for i in range(len(dataframe))]
        
    # Create subplot
    ax = plt.subplot(111)
    
    # Plot the 'Percentage' bar horizontally
    ax.barh(
        dataframe['Group'], 
        dataframe['Percentage'], 
        color=colors,
        edgecolor='white',
        alpha=fill_transparency
    )
    
    # Add the 'Remaining' bar horizontally behind the 'Percentage' bar
    ax.barh(
        dataframe['Group'], 
        dataframe['Remaining'], 
        left=dataframe['Percentage'], 
        color=background_color, 
        edgecolor='white',
    )
    
    # Set the x-axis to show percentages
    plt.xticks(np.arange(0, 101, 10))
    
    # Limit the x-axis to 100%
    plt.xlim(0, 100)
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(group_column_name, 40, break_long_words=False))  # String wrap the variable name
    plt.ylabel(wrapped_variable_name, fontsize=10, color="#262626")
    
    # Format and wrap y axis tick labels using textwrap
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label, 40, break_long_words=False)) for label in dataframe[group_column_name].tolist()]
    plt.yticks(
        dataframe['Group'], 
        wrapped_y_tick_labels, 
        fontsize=10, 
        color="#262626"
    )
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
    # Change x-axis colors to "#666666"
    plt.tick_params(axis='x', colors="#666666")
    
    # Remove the top, right, and left spines
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

