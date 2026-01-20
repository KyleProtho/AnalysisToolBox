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
    Simulate continuous variables following a Normal (Gaussian) distribution.

    This function conducts Monte Carlo simulations to model outcomes that are 
    symmetrically distributed around a mean value, where most observations cluster 
    near the center and the probability of extreme values tapers off in both directions. 
    The Normal distribution is fundamental to the Central Limit Theorem and is used 
    extensively for modeling physical, social, and financial phenomena.

    Normal distribution simulations are essential for:
      * Finance: Modeling the historical distribution of portfolio or index returns over time.
      * Healthcare: Simulating physiological measurements like blood pressure, height, or weight in a population.
      * Quality Engineering: Modeling industrial variability and tolerances in manufacturing component dimensions.
      * Intelligence Analysis: Modeling the circular error probability (CEP) or geolocation error for sensors.
      * Logistics & Supply Chain: Estimating the distribution of transit times for highly standardized delivery routes.
      * Environmental Science: Simulating annual fluctuations in average temperatures or rainfall patterns.
      * Human Resources: Modeling standardized test results or performance appraisal scores across a large workforce.
      * Engineering: Simulating the distribution of mechanical stress loads on a structural component during operation.

    The function allows the standard deviation to be explicitly provided or estimated 
    automatically from a provided range (min/max) using the range rule of thumb.

    Parameters
    ----------
    expected_outcome : float, optional
        The mean or average value of the distribution (Loc). Defaults to 0.
    standard_deviation_of_outcome : float, optional
        The standard deviation representing the spread of the data (Scale). 
        If None, it must be estimated from `min_max_of_outcome`. Defaults to None.
    min_max_of_outcome : list, optional
        A list of length 2 [min, max] used to estimate the standard deviation (range/3.29).
        Ignored if `standard_deviation_of_outcome` is provided. Defaults to None.
    number_of_trials : int, optional
        The number of stochastic simulations (Monte Carlo trials) to run. Defaults to 10000.
    return_format : str, optional
        The format of the returned data: 'dataframe' (pd.DataFrame) or 'array' (np.ndarray).
        Defaults to 'dataframe'.
    simulated_variable_name : str, optional
        The label for the simulated variable in the output and plot. Defaults to 'Simulated Outcome'.
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
        Whether to display the mean value as a vertical dashed line. Defaults to True.
    show_median : bool, optional
        Whether to display the median value as a vertical dotted line. Defaults to True.
    title_for_plot : str, optional
        The main title for the distribution plot. Defaults to "Simulation Results".
    subtitle_for_plot : str, optional
        The descriptive subtitle for the plot. Defaults to "Showing the distribution of the outcome".
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
        The simulated outcomes across all Monte Carlo runs.

    Examples
    --------
    # Healthcare: Simulating adult heights (Mean 170cm, SD 8cm)
    height_sim = SimulateNormallyDistributedOutcome(
        expected_outcome=170,
        standard_deviation_of_outcome=8,
        simulated_variable_name='Height (cm)',
        title_for_plot='Population Height Simulation'
    )

    # Manufacturing: Estimating component width using known range [9.8, 10.2]
    part_sim = SimulateNormallyDistributedOutcome(
        expected_outcome=10.0,
        min_max_of_outcome=[9.8, 10.2],
        fill_color="#2980b9"
    )
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

