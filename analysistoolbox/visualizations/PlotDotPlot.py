# Load packages
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotDotPlot(dataframe,
                categorical_column_name,
                value_column_name,
                group_column_name,
                # Dot formatting arguments
                color_palette="Paired",
                dot_size=1.5,
                dot_alpha=1,
                connect_dots=True,
                connect_line_color="#666666",
                connect_line_alpha=0.4,
                connect_line_width=0.75,
                # Plot formatting arguments
                display_order_list=None,
                figure_size=(10, 6),
                show_legend=True,
                # Text formatting arguments
                title_for_plot=None,
                subtitle_for_plot=None,
                caption_for_plot=None,
                data_source_for_plot=None,
                title_y_indent=1.15,
                subtitle_y_indent=1.1,
                caption_y_indent=-0.15,
                decimal_places_for_data_label=1,
                data_label_fontsize=11):
    """
    Plot a dot plot with optional connecting lines.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        categorical_column_name (str): The name of the categorical column in the dataframe.
        value_column_name (str): The name of the value column in the dataframe.
        group_column_name (str): The name of the group column in the dataframe.
        color_palette (str, optional): The color palette to use for the plot. Defaults to "Paired".
        dot_size (float, optional): The size of the dots in the plot. Defaults to 1.5.
        dot_alpha (float, optional): The transparency of the dots in the plot. Defaults to 1.
        connect_dots (bool, optional): Whether to connect the dots with lines. Defaults to True.
        connect_line_color (str, optional): The color of the connecting lines. Defaults to "#666666".
        connect_line_alpha (float, optional): The transparency of the connecting lines. Defaults to 0.4.
        connect_line_width (float, optional): The width of the connecting lines. Defaults to 0.75.
        display_order_list (list, optional): The order in which the categories should be displayed. 
            If not provided, the categories will be sorted by the value column in descending order. Defaults to None.
        figure_size (tuple, optional): The size of the plot figure. Defaults to (10, 6).
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        title_y_indent (float, optional): The y-axis indentation for the title. Defaults to 1.15.
        subtitle_y_indent (float, optional): The y-axis indentation for the subtitle. Defaults to 1.1.
        caption_y_indent (float, optional): The y-axis indentation for the caption. Defaults to -0.15.
        decimal_places_for_data_label (int, optional): The number of decimal places to round the data labels. Defaults to 1.
        data_label_fontsize (int, optional): The fontsize of the data labels. Defaults to 11.

    Raises:
        ValueError: If the display_order_list does not contain all of the categories in the dataframe.

    Returns:
        None
    """
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[categorical_column_name].unique())):
            raise ValueError("display_order_list must contain all of the categories in the dataframe.")
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=False)[categorical_column_name]
        display_order_list = display_order_list.unique()
    
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=figure_size)
    
    # If connect_dots is True, create horizontal lines connecting the dots
    if connect_dots == True:
        for i in range(len(display_order_list)):
            # Get the x and y coordinates for the dots
            x_coordinates = dataframe[dataframe[categorical_column_name] == display_order_list[i]][value_column_name]
            y_coordinates = dataframe[dataframe[categorical_column_name] == display_order_list[i]][categorical_column_name]
            
            # Get the x and y coordinates for the lines
            x_line_coordinates = [x_coordinates.min(), x_coordinates.max()]
            y_line_coordinates = [y_coordinates.min(), y_coordinates.max()]
            
            # Plot the lines
            plt.plot(
                x_line_coordinates, 
                y_line_coordinates, 
                color=connect_line_color, 
                alpha=connect_line_alpha, 
                linestyle='dashed', 
                linewidth=connect_line_width
            )
    
    # Create pointplot
    sns.pointplot(
        data=dataframe,
        x=value_column_name, 
        y=categorical_column_name, 
        hue=group_column_name,
        order=display_order_list,
        join=False,
        palette=color_palette, 
        markers='o', 
        scale=dot_size,
        ax=ax
    )
    
    # Remove legend if show_legend is False
    if show_legend == False:
        ax.legend_.remove()
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(categorical_column_name, 40, break_long_words=False)) 
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
    for p, label in zip(ax.patches, abs_values):
        ax.annotate(
            label,
            (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
            ha='center',
            va='center',
            fontsize=data_label_fontsize,
            color='#262626',
            xytext=(0, 0),
            textcoords='offset points'
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

