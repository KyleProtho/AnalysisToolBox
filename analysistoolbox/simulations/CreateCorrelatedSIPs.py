# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def CreateCorrelatedSIPs(mean_of_variable_1,
                         std_of_variable_1,
                         mean_of_variable_2,
                         std_of_variable_2,
                         correlation,
                         number_of_samples=10000,
                         variable_1_name='Variable 1',
                         variable_2_name='Variable 2',
                         # Printing arguments
                         print_simulation_result_summary=False,
                         # Scatterplot arguments
                         show_scatterplot=True,
                         dot_fill_color="#999999",
                         title_for_plot="Simulation Results",
                         subtitle_for_plot="Showing the correlation between two variables",
                         caption_for_plot=None,
                         data_source_for_plot=None,
                         x_indent=-0.128,
                         title_y_indent=1.125,
                         subtitle_y_indent=1.05,
                         caption_y_indent=-0.3):
    """
    This function creates a scatterplot of two variables with a specified correlation.

    Args:
        mean_of_variable_1 (float): The mean of the first variable.
        std_of_variable_1 (float): The standard deviation of the first variable.
        mean_of_variable_2 (float): The mean of the second variable.
        std_of_variable_2 (float): The standard deviation of the second variable.
        correlation (float): The correlation between the two variables.
        number_of_samples (int, optional): The number of samples to generate. Defaults to 10000.
        variable_1_name (str, optional): The name of the first variable. Defaults to 'Variable 1'.
        variable_2_name (str, optional): The name of the second variable. Defaults to 'Variable 2'.
        show_scatterplot (bool, optional): Whether to show the scatterplot. Defaults to True.
        dot_fill_color (str, optional): The color of the dots in the scatterplot. Defaults to '#999999'.
        title_for_plot (str, optional): The title of the plot. Defaults to 'Simulation Results'.
        subtitle_for_plot (str, optional): The subtitle of the plot. Defaults to 'Showing the correlation between two variables'.
        caption_for_plot (str, optional): The caption of the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source of the plot. Defaults to None.
        x_indent (float, optional): The x-indent of the plot. Defaults to -0.128.
        title_y_indent (float, optional): The y-indent of the title. Defaults to 1.125.
        subtitle_y_indent (float, optional): The y-indent of the subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent of the caption. Defaults to -0.3.
    """
    
    # Create a z-scored variable for the first variable
    variable_1 = np.random.normal(
        loc=0, 
        scale=1, 
        size=number_of_samples
    )

    # Create a z-scored variable for the second variable
    variable_2 = np.random.normal(
        loc=0, 
        scale=1, 
        size=number_of_samples
    )

    # If correlation is provided, force the correlation between the two variables
    if correlation is not None:
        variable_2 = variable_1 * correlation + (1 - correlation**2) ** 0.5 * variable_2

    # Convert the z-scored variables to the original scale
    variable_1 = variable_1 * std_of_variable_1 + mean_of_variable_1
    variable_2 = variable_2 * std_of_variable_2 + mean_of_variable_2

    # Create a dataframe
    dataframe = pd.DataFrame({
        variable_1_name: variable_1,
        variable_2_name: variable_2
    })

    # Calculate the correlation between the two variables
    simulated_correlation = dataframe[variable_1_name].corr(dataframe[variable_2_name])
    if print_simulation_result_summary:
        print("Correlation between the two variables in the simulated data: ", simulated_correlation)

    # Show the summary statistics of the dataframe
    simulation_summary = dataframe.describe()
    if print_simulation_result_summary:
        print("Summary statistics of the simulated data:")
        print(simulation_summary)

    # If scatterplot requested, generate plot
    if show_scatterplot:
        ax = sns.scatterplot(
            data=dataframe, 
            x=variable_1_name,
            y=variable_2_name,
            alpha=0.5,
            color=dot_fill_color
        )

        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
    
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
        ax.yaxis.set_label_coords(-0.1, 0.99)
        ax.yaxis.set_label_text(
            textwrap.fill(variable_2_name, 30, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right'
        )

        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.99, -0.1)
        ax.xaxis.set_label_text(
            textwrap.fill(variable_1_name, 30, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right'
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

    # Return the dataframe
    return dataframe
