# Load packages
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import textwrap

# Delcare function
def SimulateNormallyDistributedOutcome(expected_outcome=0,
                                       standard_deviation_of_outcome=None,
                                       min_max_of_outcome=None,
                                       # Simulation parameters
                                       number_of_trials=10000,
                                       return_format='dataframe',
                                       simulated_variable_name='Simulated Outcome',
                                       random_seed=412,
                                       # Plotting parameters
                                       plot_simulation_results=True,
                                       fill_color="#999999",
                                       fill_transparency=0.6,
                                       figure_size=(8, 6),
                                       show_mean=True,
                                       show_median=True,
                                       # Text formatting arguments
                                       title_for_plot="Simulation Results",
                                       subtitle_for_plot="Showing the distribution of the outcome",
                                       caption_for_plot=None,
                                       data_source_for_plot=None,
                                       show_y_axis=False,
                                       title_y_indent=1.1,
                                       subtitle_y_indent=1.05,
                                       caption_y_indent=-0.15):
    """
    Simulates normally distributed outcomes based on the specified parameters.
    The normal distribution is a continuous probability distribution that is symmetrical around its mean, most of the observations cluster around the central peak, and the probabilities for values further away from the mean taper off equally in both directions. 
    Extreme values in both tails of the distribution are similarly unlikely.
    Conditions:
    - Continuous data
    - Unbounded distribution
    - Outliers are minimal

    Args:
        expected_outcome (float): The expected value of the outcome. Defaults to 0.
        standard_deviation_of_outcome (float): The standard deviation of the outcome. If not specified, it is estimated using min_max_of_outcome. Defaults to None.
        min_max_of_outcome (list): A list of length 2 specifying the minimum and maximum values of the outcome. If specified, standard_deviation_of_outcome is ignored. Defaults to None.
        number_of_trials (int): The number of trials to simulate. Defaults to 10000.
        return_format (str): The format in which to return the simulation results. Must be either 'dataframe' or 'array'. Defaults to 'dataframe'.
        simulated_variable_name (str): The name of the simulated variable. Defaults to 'Simulated Outcome'.
        random_seed (int): The random seed to use for replicability. Defaults to 412.
        plot_simulation_results (bool): Whether to plot the simulation results. Defaults to True.
        fill_color (str): The color to use for the histogram fill. Defaults to "#999999".
        fill_transparency (float): The transparency of the histogram fill. Defaults to 0.6.
        figure_size (tuple): The size of the plot figure. Defaults to (8, 6).
        show_mean (bool): Whether to show the mean on the plot. Defaults to True.
        show_median (bool): Whether to show the median on the plot. Defaults to True.
        title_for_plot (str): The title of the plot. Defaults to "Simulation Results".
        subtitle_for_plot (str): The subtitle of the plot. Defaults to "Showing the distribution of the outcome".
        caption_for_plot (str): The caption of the plot. Defaults to None.
        data_source_for_plot (str): The data source of the plot. Defaults to None.
        show_y_axis (bool): Whether to show the y-axis on the plot. Defaults to False.
        title_y_indent (float): The y-indent of the plot title. Defaults to 1.1.
        subtitle_y_indent (float): The y-indent of the plot subtitle. Defaults to 1.05.
        caption_y_indent (float): The y-indent of the plot caption. Defaults to -0.15.

    Returns:
        pandas.DataFrame or numpy.ndarray: The simulated outcomes in the specified format.
    """
    
    # Ensure arguments are valid
    if standard_deviation_of_outcome is not None and standard_deviation_of_outcome < 0:
        raise ValueError("Please make sure that your standard_deviation_of_outcome argument is greater than or equal to 0.")
    
    # Ensure that return_format is either 'dataframe' or 'array'
    if return_format not in ['dataframe', 'array']:
        raise ValueError("return_format must be either 'dataframe' or 'array'.")
    
    # Ensure that min_max_of_outcome is either None or a list of length 2
    if min_max_of_outcome is not None:
        if len(min_max_of_outcome) != 2:
            raise ValueError("If specified, min_max_of_outcome must be a list of length 2.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
    
    # If standard_deviation_of_outcome and min_max_of_outcome are not None, print a warning
    if standard_deviation_of_outcome is not None and min_max_of_outcome is not None:
        print("Warning: Both standard_deviation_of_outcome and min_max_of_outcome were specified. Ignoring min_max_of_outcome.")
    
    # If standard_deviation_of_outcome is not specified, estimate it using min_max_of_outcome
    if standard_deviation_of_outcome is None:
        if min_max_of_outcome is None:
            raise ValueError("Please specify either standard_deviation_of_outcome or min_max_of_outcome.")
        else:
            # This formula is known as the "range rule" and it is based on the empirical finding that the range 
            # of a normal distribution is typically equal to 3.29 times its standard deviation.
            standard_deviation_of_outcome = (min_max_of_outcome[1] - min_max_of_outcome[0]) / 3.29  
        
    # Simulate normally distributed outcome
    list_sim_results = np.random.normal(
        loc=expected_outcome,
        scale=standard_deviation_of_outcome,
        size=number_of_trials
    )
    
    # Create dataframe of simulation results
    df_simulation = pd.DataFrame(list_sim_results,
                                 columns=[simulated_variable_name])
    
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

