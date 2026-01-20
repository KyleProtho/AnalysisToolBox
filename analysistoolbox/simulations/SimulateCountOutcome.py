# Load packages
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import textwrap

# Declare function
def SimulateCountOutcome(expected_count,
                         # Simulation parameters
                         number_of_trials=10000,
                         return_format='dataframe',
                         simulated_variable_name='Count',
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
                         subtitle_for_plot="Showing the distribution of the simulated count outcome",
                         caption_for_plot=None,
                         data_source_for_plot=None,
                         show_y_axis=False,
                         title_y_indent=1.1,
                         subtitle_y_indent=1.05,
                         caption_y_indent=-0.15):
    """
    Simulate discrete event counts over a fixed interval (Poisson Distribution).

    This function conducts Monte Carlo simulations to model the number of independent 
    events occurring within a fixed period of time or space, given a known average rate 
    (expected count). This is a classic Poisson distribution model, widely used for 
    modeling "arrival" processes and rare events where the probability of an event 
    is proportional to the interval length.

    Poisson count simulations are essential for:
      * Epidemiology: Modeling the daily number of new disease cases in a specific region.
      * Healthcare: Simulating patient arrivals at an emergency department per hour.
      * Intelligence Analysis: Modeling the frequency of specific signal detections per day.
      * Cybersecurity: Estimating the number of malicious port scans or log-in attempts per minute.
      * Supply Chain: Projecting the daily volume of customer orders at a warehouse.
      * Infrastructure: Simulating vehicle throughput at a highway intersection per hour.
      * Finance: Estimating the frequency of high-value transactions or system alerts.
      * Quality Control: Modeling the number of surface defects per square meter of manufactured material.

    The function uses the `numpy.random.poisson` generator for robust stochastic 
    sampling and includes automated plotting for distribution analysis.

    Parameters
    ----------
    expected_count : float
        The average number of events expected to occur in the interval (Lambda).
    number_of_trials : int, optional
        The number of stochastic simulations (Monte Carlo trials) to run. Defaults to 10000.
    return_format : str, optional
        The format of the returned data: 'dataframe' (pd.DataFrame) or 'array' (np.ndarray).
        Defaults to 'dataframe'.
    simulated_variable_name : str, optional
        The label for the simulated count variable in the output and plot. Defaults to 'Count'.
    random_seed : int, optional
        The seed for the random number generator to ensure replicability. Defaults to 412.
    plot_simulation_results : bool, optional
        Whether to display a histogram of the simulation outcomes. Defaults to True.
    fill_color : str, optional
        The hex color code for the histogram bars. Defaults to "#999999".
    fill_transparency : float, optional
        The transparency level (0-1) for the histogram plot. Defaults to 0.6.
    figure_size : tuple, optional
        The size of the plot figure in inches (width, height). Defaults to (8, 6).
    show_mean : bool, optional
        Whether to display the mean event count as a vertical dashed line. Defaults to True.
    show_median : bool, optional
        Whether to display the median event count as a vertical dotted line. Defaults to True.
    title_for_plot : str, optional
        The main title for the distribution plot. Defaults to "Simulation Results".
    subtitle_for_plot : str, optional
        The descriptive subtitle for the plot. Defaults to "Showing the distribution of the simulated count outcome".
    caption_for_plot : str, optional
        Optional caption text displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Optional data source identification text. Defaults to None.
    show_y_axis : bool, optional
        Whether to display the frequency/density scale on the y-axis. Defaults to False.
    title_y_indent : float, optional
        Vertical position for the title text. Defaults to 1.1.
    subtitle_y_indent : float, optional
        Vertical position for the subtitle text. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical position for the caption text. Defaults to -0.15.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The simulated counts of events across all trials.

    Examples
    --------
    # Healthcare: Simulating hourly ER arrivals (average of 8 per hour)
    er_arrivals = SimulateCountOutcome(
        expected_count=8.0,
        simulated_variable_name='Patient Arrivals',
        title_for_plot='Simulated ER Throughput'
    )

    # Supply Chain: Predicting daily order counts (average of 150 orders)
    order_sim = SimulateCountOutcome(
        expected_count=150,
        number_of_trials=5000,
        fill_color="#2980b9"
    )
    """
    
    # Ensure arguments are valid
    if expected_count <= 0:
        raise ValueError("Please make sure that your expected_count_of_events argument is a positive whole number.")
    
    # Ensure that return_format is either 'dataframe' or 'array'
    if return_format not in ['dataframe', 'array']:
        raise ValueError("return_format must be either 'dataframe' or 'array'.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
        
    # Simulate count outcome
    list_sim_results = np.random.poisson(lam=expected_count,
                                         size=number_of_trials)
    
    # Convert results to a dataframe
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
            binwidth=1
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

