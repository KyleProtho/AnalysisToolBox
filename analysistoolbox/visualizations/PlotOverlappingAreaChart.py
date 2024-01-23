# Load packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotOverlappingAreaChart(dataframe,
                             value_column_name,
                             variable_column_name,
                             time_column_name,
                             # Plot formatting arguments
                             figure_size=(8, 5),
                             # Line formatting arguments
                             line_alpha=0.9,
                             color_palette="Paired",
                             # Area formatting arguments
                             fill_alpha=0.8,
                             label_variable_on_area=True,
                             label_font_size=12,
                             label_font_color="#FFFFFF",
                             # Text formatting arguments
                             number_of_x_axis_ticks=None,
                             x_axis_tick_rotation=None,
                             title_for_plot=None,
                             subtitle_for_plot=None,
                             caption_for_plot=None,
                             data_source_for_plot=None,
                             x_indent=-0.127,
                             title_y_indent=1.125,
                             subtitle_y_indent=1.05,
                             caption_y_indent=-0.3,
                             # Plot saving arguments
                             filepath_to_save_plot=None):
    """
    Plots an overlapping area chart for a time series from a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data to be plotted.
        value_column_name (str): The name of the column in the dataframe that contains the values to be plotted.
        variable_column_name (str): The name of the column in the dataframe to group data by.
        time_column_name (str): The name of the column in the dataframe that contains the time data.
        figure_size (tuple, optional): The size of the figure for the plot. Defaults to (8, 5).
        line_alpha (float, optional): The transparency of the line in the plot. Defaults to 0.9.
        color_palette (str, optional): The color palette to use for the plot. Defaults to "Paired".
        fill_alpha (float, optional): The transparency of the area under the line plot. Defaults to 0.8.
        label_variable_on_area (bool, optional): Whether to label the variables on the area plot. Defaults to True.
        label_font_size (int, optional): The font size of the variable labels. Defaults to 12.
        label_font_color (str, optional): The font color of the variable labels. Defaults to "#FFFFFF".
        number_of_x_axis_ticks (int, optional): The number of ticks on the x-axis. Defaults to None.
        x_axis_tick_rotation (int, optional): The rotation of the x-axis ticks. Defaults to None.
        title_for_plot (str, optional): The title for the plot. Defaults to None.
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        x_indent (float, optional): The x-indent for the plot text. Defaults to -0.127.
        title_y_indent (float, optional): The y-indent for the plot title. Defaults to 1.125.
        subtitle_y_indent (float, optional): The y-indent for the plot subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent for the plot caption. Defaults to -0.3.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Order the dataframe by the mean of the value column for each variable
    averages = dataframe.groupby(variable_column_name).mean().reset_index()
    averages = averages.sort_values(by=value_column_name, ascending=False)
    dataframe[variable_column_name] = pd.Categorical(dataframe[variable_column_name], categories=averages[variable_column_name], ordered=True)
    dataframe = dataframe.sort_values(by=[variable_column_name])
    
    # Use Seaborn to create a line plot
    sns.lineplot(
        data=dataframe,
        x=time_column_name,
        y=value_column_name,
        hue=variable_column_name,
        palette=color_palette,
        alpha=line_alpha,
        ax=ax,
        legend=None if label_variable_on_area else "brief" # If label_variable_on_area is False, include a legend
    )
    
    # Create sns color palette
    sns_color_palette = sns.color_palette(color_palette)
    
    # Fill the area under the line plot for each variable
    for variable in dataframe[variable_column_name].unique():
        # Get the data for the variable
        variable_data = dataframe[dataframe[variable_column_name] == variable]
        
        # Fill the area under the line plot for the variable
        ax.fill_between(
            x=variable_data[time_column_name],
            y1=variable_data[value_column_name],
            y2=0,
            alpha=fill_alpha,
            color=sns_color_palette[dataframe[variable_column_name].unique().tolist().index(variable)] # Get the color of the variable from the sns color palette
        )
        
    # Add variable labels to the area under the line plot, if requested
    if label_variable_on_area:
        for variable in dataframe[variable_column_name].unique():
            # Get the data for the variable
            variable_data = dataframe[dataframe[variable_column_name] == variable]
            
            # Filter to records where the value is greater than 0
            variable_data = variable_data[variable_data[value_column_name] > 0]
        
            # Get the maximum value for the variable
            max_value = variable_data[value_column_name].max()
            
            # Add the variable label to the area under the line plot
            ax.text(
                x=variable_data[time_column_name].iloc[int(len(variable_data[time_column_name])/2)],
                y=max_value*0.4,
                s=variable,
                verticalalignment="bottom",
                # fontname="Arial",
                fontsize=label_font_size,
                fontweight="bold",
                color=label_font_color # Get the color of the last line in the plot
            )
    
    # Remove top and right spines, and set bottom and left spines to gray
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    
    # Set the number of ticks on the x-axis
    if number_of_x_axis_ticks is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(number_of_x_axis_ticks))
        
    # Rotate x-axis tick labels
    if x_axis_tick_rotation is not None:
        plt.xticks(rotation=x_axis_tick_rotation)

    # Format tick labels to be Arial, size 9, and color #666666
    ax.tick_params(
        which='major',
        labelsize=9,
        color='#666666'
    )
    
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
    
    # Word wrap the subtitle without splitting words
    if subtitle_for_plot != None:   
        subtitle_for_plot = textwrap.fill(subtitle_for_plot, 110, break_long_words=False)
        
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
    
    # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
    ax.yaxis.set_label_coords(-0.1, 0.84)
    ax.yaxis.set_label_text(
        value_column_name,
        # fontname="Arial",
        fontsize=10,
        color="#666666"
    )
    
    # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
    ax.xaxis.set_label_coords(0.9, -0.1)
    ax.xaxis.set_label_text(
        time_column_name,
        # fontname="Arial",
        fontsize=10,
        color="#666666"
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
    
