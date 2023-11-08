# Load packages
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import textwrap

# Declare function
def SimulateCountUntilFirstSuccess(probability_of_success,
                                   # Simulation parameters
                                   number_of_trials=10000,
                                   simulated_variable_name='Count Until First Success',
                                   random_seed=412,
                                   return_format='dataframe',
                                   # Plotting parameters
                                   plot_simulation_results=True,
                                   fill_color="#999999",
                                   fill_transparency=0.6,
                                   figure_size=(8, 6),
                                   show_mean=True,
                                   show_median=True,
                                   # Text formatting arguments
                                   title_for_plot="Simulation Results",
                                   subtitle_for_plot="Showing the distribution of the count until first success",
                                   caption_for_plot=None,
                                   data_source_for_plot=None,
                                   show_y_axis=False,
                                   title_y_indent=1.1,
                                   subtitle_y_indent=1.05,
                                   caption_y_indent=-0.15):
    """
    Simulate the count until the first success using a negative binomial distribution.
    A negative binomial distribution can be used to describe the number of successes r - 1
    and x failures in x + r -1 trials, until you have a success on the x + rth trial. 
    Rephrased, this models the number of failures (x) you would have to see before you see a 
    certain number of successes (r).
    Conditions:
    - Count of discrete events
    - The events CAN be non-independent, implying that events can influence or cause other events
    - Variance can exceed the mean

    Args:
        probability_of_success (float): The probability of success for each trial.
        number_of_trials (int, optional): The number of trials to simulate. Defaults to 10000.
        simulated_variable_name (str, optional): The name of the simulated variable. Defaults to 'Count Until First Success'.
        random_seed (int, optional): The random seed for replicability. Defaults to 412.
        return_format (str, optional): The format of the output. Either 'dataframe' or 'array'. Defaults to 'dataframe'.
        plot_simulation_results (bool, optional): Whether to plot the simulation results. Defaults to True.
        fill_color (str, optional): The color to fill the histogram bars with. Defaults to "#999999".
        fill_transparency (float, optional): The transparency of the histogram bars. Defaults to 0.6.
        figure_size (tuple, optional): The size of the plot figure. Defaults to (8, 6).
        show_mean (bool, optional): Whether to show the mean on the plot. Defaults to True.
        show_median (bool, optional): Whether to show the median on the plot. Defaults to True.
        title_for_plot (str, optional): The title of the plot. Defaults to "Simulation Results".
        subtitle_for_plot (str, optional): The subtitle of the plot. Defaults to "Showing the distribution of the count until first success".
        caption_for_plot (str, optional): The caption of the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source of the plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis on the plot. Defaults to False.
        title_y_indent (float, optional): The y-indent of the plot title. Defaults to 1.1.
        subtitle_y_indent (float, optional): The y-indent of the plot subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent of the plot caption. Defaults to -0.15.

    Returns:
        pandas.DataFrame or numpy.ndarray: The simulated count until the first success.
    """
    
    # Ensure probability_of_success is between 0 and 1
    if probability_of_success > 1 or probability_of_success < 0:
        raise ValueError("Please change your probability_of_success argument -- it must be greater than 0 and less than 1.")
    
    # Ensure that return_format is either 'dataframe' or 'array'
    if return_format not in ['dataframe', 'array']:
        raise ValueError("return_format must be either 'dataframe' or 'array'.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)

    # Simulate count until first success
    list_sim_results = []
    for i in range(0, number_of_trials):
        is_success = False
        event_count = 0
        while is_success == False:
            event_count += 1
            sim_result = random.choices(population = [0, 1],
                                        weights = [1-probability_of_success, probability_of_success])
            sim_result = sum(sim_result)
            if sim_result == 1:
                list_sim_results.append(event_count)
                is_success = True
            else:
                continue
    df_simulation = pd.DataFrame(list_sim_results,
                                 columns = [simulated_variable_name])
    
    # Generate plot if user requests it
    if plot_simulation_results == True:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=df_simulation,
            x=simulated_variable_name,
            color=fill_color,
            alpha=fill_transparency,
        )
        
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
        
        # Remove the x-axis label
        ax.set_xlabel(None)
        
        # Show the mean if requested
        if show_mean:
            # Calculate the mean
            mean = df_simulation[simulated_variable_name].mean()
            # Show the mean as a vertical line with a label
            ax.axvline(
                x=mean,
                ymax=0.97-.02,
                color="#262626",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5
            )
            ax.text(
                x=mean, 
                y=plt.ylim()[1] * 0.97, 
                s='Mean: {:.2f}'.format(mean),
                horizontalalignment='center',
                fontname="Arial",
                fontsize=9,
                color="#262626",
                alpha=0.75
            )
        
        # Show the median if requested
        if show_median:
            # Calculate the median
            median = df_simulation[simulated_variable_name].median()
            # Show the median as a vertical line with a label
            ax.axvline(
                x=median,
                ymax=0.90-.02,
                color="#262626",
                linestyle=":",
                linewidth=1.5,
                alpha=0.5
            )
            ax.text(
                x=median,
                y=plt.ylim()[1] * .90,
                s='Median: {:.2f}'.format(median),
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
        
    # Return the metalog distribution
    if return_format == 'dataframe':
        return df_simulation
    else:
        return np.array(list_sim_results)

