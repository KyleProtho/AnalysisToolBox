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
    Simulate the number of independent trials required to achieve the first success.

    This function conducts Monte Carlo simulations to model the waiting time (expressed 
    as the count of trials) until a specific success event occurs, given a constant 
    probability of success per trial. Technically, this models a Geometric distribution 
    (a special case of the Negative Binomial distribution where r=1), which is vital 
    for resource planning and "wait-time" risk analysis.

    Geometric wait-time simulations are essential for:
      * Intelligence Analysis: Modeling the number of days or collection attempts until a high-value signal is detected.
      * Healthcare: Simulating the number of treatment rounds or diagnostic tests until a patient achieves remission or a definitive result.
      * Sales & Marketing: Estimating the number of cold calls or touchpoints required to secure a first-time customer.
      * Quality Engineering: Predicting the number of production units inspected until the first defective item is identified.
      * Cybersecurity: Modeling the number of brute-force login attempts required to breach a test account during a security audit.
      * Human Resources: Estimating the number of candidates screened or interviewed until a qualified hire is made.
      * Engineering: Simulating the number of cycles or operations until the first instance of component failure or fatigue.
      * Scientific Research: Predicting the number of experimental trials required until a specific reaction or observation occurs.

    The function iteratively simulates Bernoulli trials until the first "success" is 
    recorded for each Monte Carlo trial, aggregating the results into a distribution 
    for analysis.

    Parameters
    ----------
    probability_of_success : float
        The constant probability of success for each individual trial (must be between 0 and 1).
    number_of_trials : int, optional
        The number of stochastic simulations (Monte Carlo trials) to run. Defaults to 10000.
    simulated_variable_name : str, optional
        The label for the simulated count variable in the output and plot. 
        Defaults to 'Count Until First Success'.
    random_seed : int, optional
        The seed for the random number generator to ensure replicability. Defaults to 412.
    return_format : str, optional
        The format of the returned data: 'dataframe' (pd.DataFrame) or 'array' (np.ndarray).
        Defaults to 'dataframe'.
    plot_simulation_results : bool, optional
        Whether to display a histogram of the simulation outcomes. Defaults to True.
    fill_color : str, optional
        The hex color code for the histogram bars. Defaults to "#999999".
    fill_transparency : float, optional
        The transparency level (0-1) for the histogram plot. Defaults to 0.6.
    figure_size : tuple, optional
        The size of the plot figure in inches (width, height). Defaults to (8, 6).
    show_mean : bool, optional
        Whether to display the mean wait time as a vertical dashed line. Defaults to True.
    show_median : bool, optional
        Whether to display the median wait time as a vertical dotted line. Defaults to True.
    title_for_plot : str, optional
        The main title for the distribution plot. Defaults to "Simulation Results".
    subtitle_for_plot : str, optional
        The descriptive subtitle for the plot. Defaults to "Showing the distribution of the count until first success".
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
        The simulated counts of trials until the first success across all Monte Carlo runs.

    Examples
    --------
    # Sales: Estimating calls needed for first conversion (10% success rate)
    calls_sim = SimulateCountUntilFirstSuccess(
        probability_of_success=0.10,
        simulated_variable_name='Calls to Close',
        title_for_plot='Sales conversion Wait Time'
    )

    # Intelligence: Days until signal detection (5% daily probability)
    detection_sim = SimulateCountUntilFirstSuccess(
        probability_of_success=0.05,
        number_of_trials=5000,
        fill_color="#e67e22"
    )
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

