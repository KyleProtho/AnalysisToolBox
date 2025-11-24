# Load packages
import os
import pandas as pd
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
import numpy as np

# Declare function
def CreateSLURPDistributionFromExponentialSmoothing(exponential_smoothing_model, 
                                                  # Simulation parameters
                                                  forecast_steps=1,
                                                  number_of_trials=10000,
                                                  prediction_interval=0.95,
                                                  lower_bound=None,
                                                  upper_bound=None,
                                                  learning_rate=.01,
                                                  term_maximum=3,
                                                  term_minimum=2,
                                                  term_for_random_sample=None,
                                                  show_summary=False,
                                                  return_format='dataframe',
                                                  # Plot parameters
                                                  show_distribution_plot=True,
                                                  figure_size=(8, 6),
                                                  fill_color="#999999",
                                                  fill_transparency=0.6,
                                                  show_mean=True,
                                                  show_median=True,
                                                  show_y_axis=False,
                                                  # Plot text parameters
                                                  title_for_plot=None,
                                                  subtitle_for_plot=None,
                                                  caption_for_plot=None,
                                                  data_source_for_plot=None,
                                                  title_y_indent=1.1,
                                                  subtitle_y_indent=1.05,
                                                  caption_y_indent=-0.15):
    """
    Create a SLURP (Simulation of Linear Uncertainty and Range Projections) distribution 
    for exponential smoothing models using prediction intervals.
    
    This function generates a distribution of future values based on the prediction 
    intervals from an exponential smoothing model, similar to how CreateSLURPDistribution
    works with linear regression models.
    
    Parameters:
    -----------
    exponential_smoothing_model : statsmodels.tsa.exponential_smoothing.ets.ETSResults
        A fitted exponential smoothing model from statsmodels
    forecast_steps : int, default=1
        Number of steps ahead to forecast (1 for one-step-ahead prediction)
    number_of_trials : int, default=10000
        Number of simulations to generate
    prediction_interval : float, default=0.95
        Prediction interval level (e.g., 0.95 for 95% interval)
    lower_bound : float, optional
        Lower bound for the distribution (if any)
    upper_bound : float, optional
        Upper bound for the distribution (if any)
    learning_rate : float, default=0.01
        Learning rate for metalog distribution fitting
    term_maximum : int, default=3
        Maximum number of terms for metalog distribution
    term_minimum : int, default=2
        Minimum number of terms for metalog distribution
    term_for_random_sample : int, optional
        Number of terms to use for random sampling (defaults to term_limit)
    show_summary : bool, default=False
        Whether to show metalog distribution summary
    return_format : str, default='dataframe'
        Return format: 'dataframe' or 'array'
    show_distribution_plot : bool, default=True
        Whether to show the distribution plot
    figure_size : tuple, default=(8, 6)
        Figure size for the plot
    fill_color : str, default="#999999"
        Color for the histogram fill
    fill_transparency : float, default=0.6
        Transparency level for the histogram fill
    show_mean : bool, default=True
        Whether to show the mean on the plot
    show_median : bool, default=True
        Whether to show the median on the plot
    show_y_axis : bool, default=False
        Whether to show the y-axis
    title_for_plot : str, optional
        Title for the plot
    subtitle_for_plot : str, optional
        Subtitle for the plot
    caption_for_plot : str, optional
        Caption for the plot
    data_source_for_plot : str, optional
        Data source for the plot
    title_y_indent : float, default=1.1
        Y position for the title
    subtitle_y_indent : float, default=1.05
        Y position for the subtitle
    caption_y_indent : float, default=-0.15
        Y position for the caption
    
    Returns:
    --------
    pandas.DataFrame or numpy.ndarray
        Distribution of simulated values
    """
    # Lazy load uncommon packages
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.exponential_smoothing.ets import ETSResults
    from .pymetalog import pymetalog
    # Create an instance of the pymetalog class
    pm = pymetalog()

    # Ensure that the exponential_smoothing_model is a statsmodels exponential smoothing model
    if not isinstance(exponential_smoothing_model, ExponentialSmoothing):
        raise ValueError("exponential_smoothing_model must be a fitted statsmodels ExponentialSmoothing model.")
    
    # Ensure that prediction_interval is between 0 and 1
    if prediction_interval <= 0 or prediction_interval >= 1:
        raise ValueError("prediction_interval must be between 0 and 1.")
    
    # Get the prediction interval using the forecast method
    forecast_result = exponential_smoothing_model.forecast(steps=forecast_steps)
    
    # Get prediction intervals
    # For exponential smoothing, we need to calculate prediction intervals manually
    # using the residual standard error
    residuals = exponential_smoothing_model.resid
    residual_std = np.std(residuals, ddof=1)  # Sample standard deviation
    
    # Calculate z-score for the prediction interval
    alpha = 1 - prediction_interval
    z_score = 1.96 if prediction_interval == 0.95 else np.abs(sm.stats.stattools.invnorm(alpha/2))
    
    # Get the forecast value (mean)
    forecast_mean = forecast_result.iloc[-1]  # Get the last forecast value
    
    # Calculate prediction interval bounds
    prediction_std = residual_std * np.sqrt(forecast_steps)
    lower_bound_value = forecast_mean - z_score * prediction_std
    upper_bound_value = forecast_mean + z_score * prediction_std
    
    # Create list of values: [lower_bound, mean, upper_bound]
    list_of_values = [lower_bound_value, forecast_mean, upper_bound_value]
    
    # If the minimum and maximum values are outside the bounds, set them to the bounds
    if lower_bound is not None and list_of_values[0] < lower_bound:
        if lower_bound < 0:
            list_of_values[0] = float(lower_bound) * 0.99999
        else:
            list_of_values[0] = float(lower_bound) * 1.00001
    if upper_bound is not None and list_of_values[2] > upper_bound:
        if upper_bound < 0:
            list_of_values[2] = float(upper_bound) * 1.00001
        else:
            list_of_values[2] = float(upper_bound) * 0.99999
        
    # If the mean is outside the bounds, slightly above/below the reference bound
    if lower_bound is not None and list_of_values[1] < lower_bound:
        if lower_bound < 0:
            list_of_values[1] = list_of_values[0] * 0.99999
        else:
            list_of_values[1] = list_of_values[0] * 1.00001
    if upper_bound is not None and list_of_values[1] > upper_bound:
        if upper_bound < 0:
            list_of_values[1] = list_of_values[2] * 1.00001
        else:
            list_of_values[1] = list_of_values[2] * 0.99999
        
    # Set the list of percentiles
    list_of_percentiles = [
        (1 - prediction_interval) / 2,  # Lower bound
        0.5,  # Middle
        1 - (1 - prediction_interval) / 2  # Upper bound
    ]
    
    # Use the prediction interval to simulate the prediction distribution
    if lower_bound is None and upper_bound is None:
        metalog_dist = pm.metalog(
            x=list_of_values,
            probs=list_of_percentiles,
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    elif lower_bound is not None and upper_bound is None:
        metalog_dist = pm.metalog(
            x=list_of_values,
            probs=list_of_percentiles,
            boundedness='sl',
            bounds=[lower_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    elif lower_bound is None and upper_bound is not None:
        metalog_dist = pm.metalog(
            x=list_of_values,
            probs=list_of_percentiles,
            boundedness='su',
            bounds=[upper_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    else:
        metalog_dist = pm.metalog(
            x=list_of_values,
            probs=list_of_percentiles,
            boundedness='b',
            bounds=[lower_bound, upper_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
        
    # Show summary of the metalog distribution
    if show_summary:
        pm.summary(m = metalog_dist)
        
    # Get the maximum number of terms used in the metalog distribution that is valid
    if term_for_random_sample is None:
        term_for_random_sample = metalog_dist.term_limit

    # Take a random sample from the metalog distribution
    arr_metalog = pm.rmetalog(
        m=metalog_dist, 
        n=number_of_trials,
        term=term_for_random_sample
    )
    
    # Create a column name for the outcome variable
    outcome_variable = 'forecast_value'
    
    # Convert the array to a dataframe
    metalog_df = pd.DataFrame(arr_metalog, columns=[outcome_variable])
    
    # Plot the metalog distribution
    if show_distribution_plot:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=metalog_df,
            x=outcome_variable,
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
            mean = metalog_df[outcome_variable].mean()
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
            median = metalog_df[outcome_variable].median()
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
        return metalog_df
    else:
        return arr_metalog
