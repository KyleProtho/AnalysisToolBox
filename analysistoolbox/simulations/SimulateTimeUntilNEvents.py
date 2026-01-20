# Load packages
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import textwrap

# Declare function
def SimulateTimeUntilNEvents(number_of_events=1,
                             expected_time_between_events=1,
                             number_of_trials=10000,
                             random_seed=412,
                             return_format='dataframe',
                             simulated_variable_name='Time Until N Events',
                             plot_simulation_results=True,
                             fill_color="#999999",
                             fill_transparency=0.6,
                             figure_size=(8, 6),
                             show_mean=True,
                             show_median=True,
                             title_for_plot="Simulation Results",
                             subtitle_for_plot="Showing the distribution of time until n events occur",
                             caption_for_plot=None,
                             data_source_for_plot=None,
                             show_y_axis=False,
                             title_y_indent=1.1,
                             subtitle_y_indent=1.05,
                             caption_y_indent=-0.15):
    """
    Simulate the total time required for a specific number of events to occur (Gamma Distribution).

    This function conducts Monte Carlo simulations to model the cumulative waiting 
    time until a predefined number of independent events take place. While the 
    Exponential distribution models the time until the *first* success, the Gamma 
    distribution generalizes this to model the time until the *Nth* success. 
    It is a critical tool for capacity planning and project duration estimation.

    Time-until-N-events simulations are essential for:
      * Healthcare: Modeling the expected time until a specialized clinic completes 'N' complex surgeries or procedures.
      * Customer Support: Simulating the total time required for a technical team to resolve 'N' high-priority tickets.
      * Intelligence Analysis: Estimating the time window needed to collect 'N' distinct intelligence artifacts for an assessment.
      * Quality Engineering: Modeling the time until a manufacturing process produces 'N' defective units for reliability testing.
      * Logistics & Warehouse: Simulating the total time until a distribution hub processes and clears 'N' scheduled shipments.
      * Project Management: Estimating the total duration of a project phase that requires the completion of 'N' sequential milestones.
      * Finance: Modeling the time required for an algorithmic system to execute 'N' blocks of automated trades under market conditions.
      * Meteorology: Predicting the likely time period until a geographic region experiences 'N' instances of an extreme weather event.

    The function uses the `numpy.random.gamma` generator where `shape` is the 
    `number_of_events` (alpha) and `scale` is the `expected_time_between_events` (theta).

    Parameters
    ----------
    number_of_events : int, optional
        The total count of events that must occur for the simulation to conclude (Shape). 
        Defaults to 1.
    expected_time_between_events : float, optional
        The average or expected time interval between individual events (Scale). 
        Defaults to 1.
    number_of_trials : int, optional
        The number of stochastic simulations (Monte Carlo trials) to run. Defaults to 10000.
    random_seed : int, optional
        The seed for the random number generator to ensure replicability. Defaults to 412.
    return_format : str, optional
        The format of the returned data: 'dataframe' (pd.DataFrame) or 'array' (np.ndarray).
        Defaults to 'dataframe'.
    simulated_variable_name : str, optional
        The label for the simulated time variable in the output and plot. 
        Defaults to 'Time Until N Events'.
    plot_simulation_results : bool, optional
        Whether to display a histogram of the simulation outcomes. Defaults to True.
    fill_color : str, optional
        The hex color code for the histogram bars. Defaults to "#999999".
    fill_transparency : float, optional
        The transparency level (0-1) for the histogram plot. Defaults to 0.6.
    figure_size : tuple, optional
        The size of the plot figure in inches (width, height). Defaults to (8, 6).
    show_mean : bool, optional
        Whether to display the mean completion time as a vertical dashed line. Defaults to True.
    show_median : bool, optional
        Whether to display the median completion time as a vertical dotted line. Defaults to True.
    title_for_plot : str, optional
        The main title for the distribution plot. Defaults to "Simulation Results".
    subtitle_for_plot : str, optional
        The descriptive subtitle for the plot. Defaults to "Showing the distribution of time until n events occur".
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
        The simulated total times across all Monte Carlo runs.

    Examples
    --------
    # Healthcare: Time until 5 surgeries are completed (Avg 4 hours each)
    surgery_sim = SimulateTimeUntilNEvents(
        number_of_events=5,
        expected_time_between_events=4.0,
        simulated_variable_name='Hours to Complete Batch',
        title_for_plot='Surgical Suite Throughput Simulation'
    )

    # Project Management: Time until 10 software modules are developed (Avg 3 days each)
    dev_sim = SimulateTimeUntilNEvents(
        number_of_events=10,
        expected_time_between_events=3,
        fill_color="#27ae60"
    )
    """
    
    # Ensure arguments are valid
    if number_of_events <= 0:
        raise ValueError("Please make sure that your number_of_events argument is greater than 0.")
    if expected_time_between_events <= 0:
        raise ValueError("Please make sure that your expected_time_between_events argument is greater than 0.")
    
    # Ensure that return_format is either 'dataframe' or 'array'
    if return_format not in ['dataframe', 'array']:
        raise ValueError("return_format must be either 'dataframe' or 'array'.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
        
    # Simulate time between events
    list_sim_results = np.random.gamma(
        shape=number_of_events,
        scale=expected_time_between_events,
        size=number_of_trials
    )
    
    # Convert results to a dataframe
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

