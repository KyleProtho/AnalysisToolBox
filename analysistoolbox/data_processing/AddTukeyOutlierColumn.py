# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def AddTukeyOutlierColumn(dataframe, 
                          value_column_name,
                          tukey_boundary_multiplier=1.5,
                          plot_tukey_outliers=False,
                          # Histogram formatting arguments
                          fill_color="#999999",
                          fill_transparency=0.6,
                          # Text formatting arguments
                          title_for_plot="Tukey Outliers",
                          subtitle_for_plot="Shows the values that are Tukey outliers.",
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          show_y_axis=False,
                          title_y_indent=1.1,
                          subtitle_y_indent=1.05,
                          caption_y_indent=-0.15,
                          # Plot formatting arguments
                          figure_size=(8, 6)):
    """
    Adds a column to the DataFrame that flags Tukey outliers.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        value_column_name (str): The name of the column to check for outliers.
        tukey_boundary_multiplier (float, optional): The number of IQRs to use for the Tukey outlier boundary. Defaults to 1.5.
        plot_tukey_outliers (bool, optional): Whether to plot the Tukey outliers. Defaults to False.
        fill_color (str, optional): The color to fill the histogram bars with. Defaults to "#999999".
        fill_transparency (float, optional): The transparency of the histogram bars. Defaults to 0.6.
        title_for_plot (str, optional): The title of the plot. Defaults to "Tukey Outliers".
        subtitle_for_plot (str, optional): The subtitle of the plot. Defaults to "Shows the values that are Tukey outliers.".
        caption_for_plot (str, optional): The caption of the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source of the plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis. Defaults to False.
        title_y_indent (float, optional): The y-indent of the title. Defaults to 1.1.
        subtitle_y_indent (float, optional): The y-indent of the subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent of the caption. Defaults to -0.15.
        figure_size (tuple, optional): The size of the plot figure. Defaults to (8, 6).

    Returns:
        pandas.DataFrame: The original DataFrame with an added column for outlier flags.
    """
    
    # Calculate the IQR
    Q1 = dataframe[value_column_name].quantile(0.25)
    Q3 = dataframe[value_column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define the outlier bounds
    lower_bound = Q1 - tukey_boundary_multiplier * IQR
    upper_bound = Q3 + tukey_boundary_multiplier * IQR

    # Flag outliers
    dataframe[f'{value_column_name} - Tukey outlier'] = ((dataframe[value_column_name] < lower_bound) | (dataframe[value_column_name] > upper_bound)).astype(int)
    
    # Plot outliers, if requested
    if plot_tukey_outliers:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=dataframe,
            x=value_column_name,
            color=fill_color,
            alpha=fill_transparency,
            hue=f'{value_column_name} - Tukey outlier',
            palette={True: "#FF0000", False: "#999999"}
        )
        
        # Remove the legend
        ax.get_legend().remove()
        
        # Remove top, left, and right spines. Set bottom spine to dark gray.
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color("#262626")
        
        # Remove the y-axis, and adjust the indent of the plot titles
        if show_y_axis == False:
            ax.axes.get_yaxis().set_visible(False)
            x_indent = 0.015
        else:
            x_indent = -0.005
        
        # Remove the y-axis label
        ax.set_ylabel(None)
        
        # Show the lower boundary as a vertical line with a label
        ax.axvline(
            x=lower_bound,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=lower_bound, 
            y=plt.ylim()[1] * 0.97, 
            s='Lower bound: {:.2f}'.format(lower_bound),
            horizontalalignment='center',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
        
        # Show the upper boundary as a vertical line with a label
        ax.axvline(
            x=upper_bound,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=upper_bound, 
            y=plt.ylim()[1] * 0.97, 
            s='Upper bound: {:.2f}'.format(upper_bound),
            horizontalalignment='center',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
        
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent,
            y=title_y_indent,
            s=title_for_plot,
            fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
        
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
        plt.xticks(fontname='Arial')
        
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
                fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
            
        # Show plot
        plt.show()
        
        # Clear plot
        plt.clf()
    
    # Return the updated dataframe
    return dataframe

