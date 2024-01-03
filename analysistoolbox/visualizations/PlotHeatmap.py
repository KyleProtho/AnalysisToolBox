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
                color_map_minimum_value=None,
                color_map_minimum_color=13,
                color_map_maximum_value=None,
                color_map_maximum_color=125,
                color_map_center_value=None,
                color_map_saturation=80,
                color_map_lightness=60,
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
                data_label_format="d",
                title_for_plot=None,
                subtitle_for_plot=None,
                caption_for_plot=None,
                data_source_for_plot=None,
                title_y_indent=1.15,
                subtitle_y_indent=1.1,
                caption_y_indent=-0.17):
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
    
    # Plot contingency table in a heatmap using seaborn
    ax = sns.heatmap(
        data=dataframe,
        cmap=sns.diverging_palette(
            color_map_minimum_color, 
            color_map_maximum_color, 
            s=color_map_saturation, 
            l=color_map_lightness,
            as_cmap=True
        ),
        center=color_map_center_value,
        vmin=color_map_minimum_value,
        vmax=color_map_maximum_value,
        annot=show_data_labels,
        fmt=data_label_format,
        linewidths=border_line_width,
        linecolor=border_line_color,
        cbar=show_legend,
        square=square_cells,
        ax=ax,
        # Bold the text in the heatmap
        annot_kws={"fontweight": data_label_fontweight,
                #    "fontname": "Arial", 
                   "fontsize": data_label_fontsize,
                   "color": data_label_color}
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
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

