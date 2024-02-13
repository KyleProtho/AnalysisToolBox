# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotSingleVariableCountPlot(dataframe,
                                categorical_column_name, 
                                # Plot formatting arguments
                                fill_color="#8eb3de",
                                color_palette=None,
                                top_n_to_highlight=None,
                                highlight_color="#b0170c",
                                fill_transparency=0.8,
                                add_rare_category_line=False,
                                rare_category_line_color='#b5b3b3',
                                rare_category_threshold=0.05,
                                figure_size=(8, 6),
                                # Text formatting arguments
                                data_label_fontsize=10,
                                data_label_padding=None,
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
    Creates a bar chart for a single categorical variable.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the categorical variable.
        categorical_column_name (str): The categorical variable to plot.
        color_palette (str, optional): The Seaborn color palette to use for the bar chart. Defaults to "Set1".
        fill_color (str, optional): The color to use for the bar chart. Defaults to None.
        top_n_to_highlight (int, optional): The number of top categories to highlight. Defaults to None.
        highlight_color (str, optional): The color to use for the highlighted categories. Defaults to "#b0170c".
        fill_transparency (float, optional): The transparency to use for the bar chart. Defaults to 0.8.
        add_rare_category_line (bool, optional): Whether to add a line to indicate the threshold for rare categories. Defaults to False.
        rare_category_line_color (str, optional): The color to use for the rare category line. Defaults to '#b5b3b3'.
        rare_category_threshold (float, optional): The threshold for rare categories. Defaults to 0.05.
        figure_size (tuple, optional): The size of the figure. Defaults to (8, 6).
        data_label_fontsize (int, optional): The font size to use for the data labels. Defaults to 10.
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        title_y_indent (float, optional): The vertical indent for the title. Defaults to 1.15.
        subtitle_y_indent (float, optional): The vertical indent for the subtitle. Defaults to 1.1.
        caption_y_indent (float, optional): The vertical indent for the caption. Defaults to -0.15.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
    
    Returns:
        None
    """
    
    # Check that the column exists in the dataframe.
    if categorical_column_name not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(categorical_column_name))
    
    # Ensure that rare category threshold is between 0 and 1
    if rare_category_threshold < 0 or rare_category_threshold > 1:
        raise ValueError("Rare category threshold must be between 0 and 1.")
    
    # Make sure that the top_n_to_highlight is a positive integer, and less than the number of categories
    if top_n_to_highlight != None:
        if top_n_to_highlight < 0 or top_n_to_highlight > len(dataframe[categorical_column_name].value_counts()):
            raise ValueError("top_n_to_highlight must be a positive integer, and less than the number of categories.")
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate bar chart
    if top_n_to_highlight != None:
        ax = sns.countplot(
            data=dataframe,
            y=categorical_column_name,
            order=dataframe[categorical_column_name].value_counts(ascending=False).index,
            palette=["#b8b8b8" if x not in dataframe[categorical_column_name].value_counts().index[:top_n_to_highlight] else highlight_color for x in dataframe[categorical_column_name].value_counts().index],
            alpha=fill_transparency
        )
    elif color_palette != None:
        ax = sns.countplot(
            data=dataframe,
            y=categorical_column_name,
            order=dataframe[categorical_column_name].value_counts(ascending=False).index,
            hue=dataframe[categorical_column_name],
            palette=color_palette,
            alpha=fill_transparency
        )
    else:
        ax = sns.countplot(
            data=dataframe,
            y=categorical_column_name,
            order=dataframe[categorical_column_name].value_counts(ascending=False).index,
            color=fill_color,
            alpha=fill_transparency
        )
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
    # Change x-axis colors to "#666666"
    ax.tick_params(axis='x', colors="#666666")
    ax.spines['top'].set_color("#666666")
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(categorical_column_name, 40, break_long_words=False))  # String wrap the variable name
    ax.set_ylabel(wrapped_variable_name)
    
    # Set x-axis title to "Count"
    ax.set_xlabel(
        "Count", 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Remove bottom, left, and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Get the maximum value of the x-axis
    if data_label_padding == None:
        max_x = dataframe[categorical_column_name].value_counts().max()
        data_label_padding = max_x * 0.025
    
    # Add data labels
    abs_values = dataframe[categorical_column_name].value_counts(ascending=False)
    rel_values = dataframe[categorical_column_name].value_counts(ascending=False, normalize=True).values * 100
    lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
    for i, p in enumerate(ax.patches):
        ax.text(
            p.get_width() + data_label_padding,
            p.get_y() + p.get_height() / 2,
            lbls[i],
            ha="left",
            va="center",
            fontsize=data_label_fontsize,
            color="#262626"
        )
    
    # Add rare category threshold line
    if add_rare_category_line:
        ax.axvline(
            x=rare_category_threshold * dataframe.shape[0],
            color=rare_category_line_color,
            alpha=0.5,
            linestyle='--',
            label='Rare category threshold'
        )
    
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

