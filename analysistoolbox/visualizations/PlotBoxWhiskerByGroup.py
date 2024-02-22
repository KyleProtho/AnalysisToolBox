# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotBoxWhiskerByGroup(dataframe,
                          value_column_name,
                          grouping_column_name_1,
                          grouping_column_name_2=None,
                          # Plot formatting arguments
                          fill_color="#8eb3de",
                          color_palette=None,
                          display_order_list=None,
                          show_legend=True,
                          # Text formatting arguments
                          title_for_plot=None,
                          subtitle_for_plot=None,
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          show_y_axis=False,
                          title_y_indent=1.1,
                          subtitle_y_indent=1.05,
                          caption_y_indent=-0.15,
                          x_indent=-0.10,
                          figure_size=(8, 6),
                          # Plot saving arguments
                          filepath_to_save_plot=None):
    """
    Create a box and whisker plot for a given outcome variable, grouped by one or two categorical variables.

    Args:
        dataframe (pandas.DataFrame): The input dataframe.
        value_column_name (str): The name of the outcome variable.
        grouping_column_name_1 (str): The name of the first group variable.
        grouping_column_name_2 (str, optional): The name of the second group variable. Defaults to None.
        fill_color (str, optional): The color to fill the box plot with. Defaults to None.
        color_palette (str, optional): The color palette to use for the box plot. Defaults to 'Set2'.
        display_order_list (list, optional): The order to display the 1st categories in the plot. Defaults to None.
        show_legend (bool, optional): Whether to show the legend. Defaults to True.
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis. Defaults to False.
        title_y_indent (float, optional): The y-indent for the title. Defaults to 1.1.
        subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.15.
        x_indent (float, optional): The x-indent for the plot. Defaults to -0.128.
        figure_size (tuple, optional): The size of the plot. Defaults to (8, 6).
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
    """
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[grouping_column_name_1].unique())):
            raise ValueError("display_order_list must contain all of the categories in the " + grouping_column_name_1 + " column of dataframe.")
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=True)[grouping_column_name_1].unique()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Generate box whisker plot
    if grouping_column_name_2 != None:
        if fill_color != None:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                hue=grouping_column_name_2, 
                color=fill_color,
                order=display_order_list,
            )
        else:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                hue=grouping_column_name_2, 
                palette=color_palette,
                order=display_order_list,
            )
    else:
        if fill_color != None:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                color=fill_color,
                order=display_order_list,
            )
        else:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                palette=color_palette,
                order=display_order_list,
            )
    
    # Remove top, and right spines. Set bottom and left spine to dark gray.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#262626")
    ax.spines['left'].set_color("#262626")
    
    # If show_legend is False, remove the legend
    if show_legend == False:
        ax.legend_.remove()
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
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
    
    # String wrap group variable 1 tick labels
    group_variable_1_tick_labels = ax.get_xticklabels()
    group_variable_1_tick_labels = [label.get_text() for label in group_variable_1_tick_labels]
    group_variable_1_tick_labels = ['\n'.join(textwrap.wrap(label, 20, break_long_words=False)) for label in group_variable_1_tick_labels]
    ax.set_xticklabels(group_variable_1_tick_labels)
    
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
    
    # Set y-axis tick label font to Arial, size 9, and color #666666
    ax.tick_params(
        axis='y',
        which='major',
        labelsize=9,
        labelcolor="#666666",
        pad=2,
        bottom=True,
        labelbottom=True
    )
    # plt.yticks(fontname='Arial')
    
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

