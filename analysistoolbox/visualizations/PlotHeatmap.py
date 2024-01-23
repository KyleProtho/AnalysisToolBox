# Load packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotHeatmap(dataframe,
                x_axis_column_name,
                y_axis_column_name,
                value_column_name,
                # Heatmap formatting arguments
                color_palette="RdYlGn",
                color_map_transparency=0.8,
                color_map_buckets=7,
                color_map_minimum_value=None,
                color_map_maximum_value=None,
                color_map_center_value=None,
                show_legend=False,
                figure_size=(8, 6),
                border_line_width=6,
                border_line_color="#ffffff",
                square_cells=False,
                # Text formatting arguments
                show_data_labels=True,
                data_label_fontweight="normal",
                data_label_color="#262626",
                data_label_fontsize=12,
                title_for_plot=None,
                subtitle_for_plot=None,
                caption_for_plot=None,
                data_source_for_plot=None,
                title_y_indent=1.15,
                subtitle_y_indent=1.1,
                caption_y_indent=-0.17,
                # Plot saving arguments
                filepath_to_save_plot=None,
                plot_dpi=300):
    """
    Plots a heatmap based on the provided dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        x_axis_column_name (str): The column name to be used as the x-axis.
        y_axis_column_name (str): The column name to be used as the y-axis.
        value_column_name (str): The column name to be used as the values for the heatmap.
        color_palette (str or list, optional): The color palette to be used for the heatmap. Defaults to "RdYlGn".
        color_map_transparency (float, optional): The transparency of the color map. Defaults to 0.8.
        color_map_buckets (int, optional): The number of color map buckets. Defaults to 7.
        color_map_minimum_value (float, optional): The minimum value for the color map. If not provided, the minimum value in the dataframe will be used.
        color_map_maximum_value (float, optional): The maximum value for the color map. If not provided, the maximum value in the dataframe will be used.
        color_map_center_value (float, optional): The center value for the color map. If not provided, the median value in the dataframe will be used.
        show_legend (bool, optional): Whether to show the legend. Defaults to False.
        figure_size (tuple, optional): The size of the figure. Defaults to (8, 6).
        border_line_width (int, optional): The width of the border lines. Defaults to 6.
        border_line_color (str, optional): The color of the border lines. Defaults to "#ffffff".
        square_cells (bool, optional): Whether to use square cells in the heatmap. Defaults to False.
        show_data_labels (bool, optional): Whether to show data labels in the heatmap. Defaults to True.
        data_label_fontweight (str, optional): The font weight of the data labels. Defaults to "normal".
        data_label_color (str, optional): The color of the data labels. Defaults to "#262626".
        data_label_fontsize (int, optional): The font size of the data labels. Defaults to 12.
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        title_y_indent (float, optional): The y-indent for the title. Defaults to 1.15.
        subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.1.
        caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.17.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
        plot_dpi (int, optional): The DPI (dots per inch) for the saved plot. Defaults to 300.

    Returns:
        None
    """
    
    # Convert dataframe to wide-form dataframe, using x_axis_column_name as the index and y_axis_column_name as the columns
    dataframe = dataframe.pivot(
        index=x_axis_column_name, 
        columns=y_axis_column_name,
        values=value_column_name
    )
    
    # Transpose dataframe
    dataframe = dataframe.transpose()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Adjust color map minimum and maximum values if they are not provided
    if color_map_minimum_value == None:
        color_map_minimum_value = dataframe.min().min()
    if color_map_maximum_value == None:
        color_map_maximum_value = dataframe.max().max()
    if color_map_center_value == None:
        color_map_center_value = dataframe.median().median()
        
    # Convert color palette to a list if it is not already a list
    if type(color_palette) != list:
        color_palette = sns.color_palette(color_palette, color_map_buckets)
        # color_palette.set_bad(color='white', alpha=color_palette_transparency)
    
    # Plot contingency table in a heatmap using seaborn
    ax = sns.heatmap(
        data=dataframe,
        cmap=color_palette,
        center=color_map_center_value,
        vmin=color_map_minimum_value,
        vmax=color_map_maximum_value,
        annot=show_data_labels,
        linewidths=border_line_width,
        linecolor=border_line_color,
        cbar=show_legend,
        square=square_cells,
        ax=ax,
        annot_kws={
            "fontweight": data_label_fontweight,
            # "fontname": "Arial", 
            "fontsize": data_label_fontsize,
            "color": data_label_color,
        }
    )
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(y_axis_column_name, 30))  # String wrap the variable name
    ax.set_ylabel(
        wrapped_variable_name, 
        fontsize=12, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in y_tick_labels]
    ax.set_yticklabels(
        wrapped_y_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Wrap x axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(x_axis_column_name, 30))  # String wrap the variable name
    ax.set_xlabel(
        wrapped_variable_name, 
        fontsize=12, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Format and wrap x axis tick labels using textwrap
    x_tick_labels = ax.get_xticklabels()
    wrapped_x_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in x_tick_labels]
    ax.set_xticklabels(
        wrapped_x_tick_labels, 
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
            dpi=plot_dpi
        )
       
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

