# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotBarChart(dataframe,
                 categorical_column_name, 
                 value_column_name,
                 # Bar formatting arguments
                 color_palette="Set1",
                 fill_color=None,
                 top_n_to_highlight=None,
                 highlight_color="#b0170c",
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
                 # Plot saving arguments
                 filepath_to_save_plot=None,
                 plot_dpi=300):
    """
    Generates a bar chart using the seaborn library.

    Args:
        dataframe (pandas dataframe): The dataframe containing the data to be plotted.
        categorical_column_name (str): The name of the column in the dataframe containing the categorical variable.
        value_column_name (str): The name of the column in the dataframe containing the value variable.
        color_palette (str or list, optional): The seaborn color palette to use for the plot. Defaults to "Set1".
        fill_color (str, optional): The color to use for the bars in the plot. If None, the color palette will be used. Defaults to None.
        top_n_to_highlight (int, optional): The number of top categories to highlight in the plot. Defaults to None.
        highlight_color (str, optional): The color to use for the highlighted categories. Defaults to "#b0170c".
        fill_transparency (float, optional): The transparency of the bars in the plot. Defaults to 0.8.
        display_order_list (list, optional): The order to display the categories in the plot. Defaults to None.
        figure_size (tuple, optional): The size of the plot figure. Defaults to (8, 6).
        title_for_plot (str, optional): The title of the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle of the plot. Defaults to None.
        caption_for_plot (str, optional): The caption of the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source of the plot. Defaults to None.
        title_y_indent (float, optional): The y-axis indent for the plot title. Defaults to 1.15.
        subtitle_y_indent (float, optional): The y-axis indent for the plot subtitle. Defaults to 1.1.
        caption_y_indent (float, optional): The y-axis indent for the plot caption. Defaults to -0.15.
        decimal_places_for_data_label (int, optional): The number of decimal places to round the data labels to. Defaults to 2.
        data_label_fontsize (int, optional): The fontsize of the data labels. Defaults to 11.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
        plot_dpi (int, optional): The DPI (dots per inch) of the saved plot. Defaults to 300.

    Returns:
        None
    """
    
    # Check that the column exists in the dataframe.
    if categorical_column_name not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(categorical_column_name))
    
    # Make sure that the top_n_to_highlight is a positive integer, and less than the number of categories
    if top_n_to_highlight != None:
        if top_n_to_highlight < 0 or top_n_to_highlight > len(dataframe[categorical_column_name].value_counts()):
            raise ValueError("top_n_to_highlight must be a positive integer, and less than the number of categories.")
        
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[categorical_column_name].unique())):
            raise ValueError("display_order_list must contain all of the categories in the dataframe.")
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=False)[categorical_column_name]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate bar chart
    if top_n_to_highlight != None:
        ax = sns.barplot(
            data=dataframe,
            y=categorical_column_name,
            x=value_column_name,
            palette=[highlight_color if x in dataframe.sort_values(value_column_name, ascending=False)[value_column_name].nlargest(top_n_to_highlight).index else "#b8b8b8" for x in dataframe.sort_values(value_column_name, ascending=False).index],
            order=display_order_list,
            alpha=fill_transparency
        )
    elif fill_color == None:
        ax = sns.barplot(
            data=dataframe,
            y=categorical_column_name,
            x=value_column_name,
            palette=color_palette,
            order=display_order_list,
            alpha=fill_transparency
        )
    else:
        ax = sns.barplot(
            data=dataframe,
            y=categorical_column_name,
            x=value_column_name,
            color=fill_color,
            order=display_order_list,
            alpha=fill_transparency
        )
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(categorical_column_name, 40, break_long_words=False))  # String wrap the variable name
    ax.set_ylabel(wrapped_variable_name)
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in y_tick_labels]
    ax.set_yticklabels(
        wrapped_y_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
    # Change x-axis colors to "#666666"
    ax.tick_params(axis='x', colors="#666666")
    ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add data labels
    abs_values = dataframe.sort_values(value_column_name, ascending=False)[value_column_name].round(decimal_places_for_data_label)
    # Create format code for data labels
    data_label_format = "{:." + str(decimal_places_for_data_label) + "f}"
    lbls = [data_label_format.format(p) for p in abs_values]
    ax.bar_label(container=ax.containers[0],
                 labels=lbls,
                 padding=5,
                 fontsize=data_label_fontsize)
        
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
            dpi=plot_dpi,
            bbox_inches="tight"
        )
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

