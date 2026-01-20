# Load packages
import os
import pandas as pd
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
import numpy as np
import statsmodels.api as sm
from scipy import stats

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
    Generate a SLURP distribution for future forecasts using a fitted Exponential Smoothing model.

    This function creates a SLURP (Simulation of Linear Uncertainty and Range Projections) 
    distribution by combining exponential smoothing forecasts with Metalog distribution 
    fitting. It calculates prediction intervals based on the model's residual standard error 
    for a specified number of future steps, and then fits a flexible Metalog distribution
    to those intervals. This allows for stochastic sampling from the predicted uncertainty 
    range, which is essential for propagating time-series uncertainty into Monte Carlo 
    simulations and probabilistic risk assessments.

    SLURP distributions from exponential smoothing are essential for:
      * Supply Chain: Modeling future inventory requirements given seasonal demand patterns.
      * Healthcare: Projecting future patient admission volumes with uncertainty for capacity planning.
      * Finance: Simulating future revenue or cash flows based on historical trends and volatility.
      * Epidemiology: Forecasting future disease incidence rates and the associated range of uncertainty.
      * Intelligence Analysis: Projecting future regional stability indices or economic indicators.
      * Energy: Simulating future power demand forecasts for grid reliability assessments.
      * Public Health: Estimating future vaccine uptake rates to optimize distribution logistics.
      * Retail: Forecasting future store traffic for staffing and resource allocation.

    The function manually calculates prediction intervals using the residual standard 
    deviation scaled by the square root of the number of forecast steps, then uses the 
    `pymetalog` library to generate a continuous distribution from the mean and interval bounds.

    Parameters
    ----------
    exponential_smoothing_model : statsmodels.tsa.exponential_smoothing.ets.ETSResults
        A fitted exponential smoothing (ETS or Holt-Winters) model from statsmodels.
    forecast_steps : int, optional
        The number of steps into the future to forecast (e.g., 1 for next day, 7 for next week).
        Defaults to 1.
    number_of_trials : int, optional
        The number of stochastic samples to generate from the uncertainty distribution.
        Defaults to 10000.
    prediction_interval : float, optional
        The confidence level for the prediction interval (between 0 and 1). Defaults to 0.95.
    lower_bound : float, optional
        A logical lower limit for the simulated values (e.g., 0 for counts/prices).
        Defaults to None.
    upper_bound : float, optional
        A logical upper limit for the simulated values. Defaults to None.
    learning_rate : float, optional
        The step length for the Metalog fitting algorithm. Defaults to 0.01.
    term_maximum : int, optional
        The maximum number of terms in the Metalog expansion (up to 9). Higher terms
        increase shape flexibility. Defaults to 3.
    term_minimum : int, optional
        The minimum number of terms allowed. Defaults to 2.
    term_for_random_sample : int, optional
        The specific number of terms used for generating samples. If None, the `term_limit`
        from the fit is used. Defaults to None.
    show_summary : bool, optional
        Whether to print the summary of the Metalog distribution coefficients. Defaults to False.
    return_format : str, optional
        The format of the output: 'dataframe' (pd.DataFrame) or 'array' (np.ndarray).
        Defaults to 'dataframe'.
    show_distribution_plot : bool, optional
        Whether to display a histogram of the generated samples. Defaults to True.
    figure_size : tuple, optional
        The size of the plot figure in inches (width, height). Defaults to (8, 6).
    fill_color : str, optional
        The hex color code for the histogram bars. Defaults to "#999999".
    fill_transparency : float, optional
        The transparency level (0-1) for the histogram plot. Defaults to 0.6.
    show_mean : bool, optional
        Whether to display the mean as a vertical dashed line on the plot. Defaults to True.
    show_median : bool, optional
        Whether to display the median as a vertical dotted line on the plot. Defaults to True.
    show_y_axis : bool, optional
        Whether to display the frequency/density scale on the y-axis. Defaults to False.
    title_for_plot : str, optional
        The main title for the distribution plot. Defaults to None.
    subtitle_for_plot : str, optional
        The descriptive subtitle for the plot. Defaults to None.
    caption_for_plot : str, optional
        Optional caption text displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Optional data source identification text. Defaults to None.
    title_y_indent : float, optional
        Vertical position for the title text. Defaults to 1.1.
    subtitle_y_indent : float, optional
        Vertical position for the subtitle text. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical position for the caption text. Defaults to -0.15.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The generated samples representing the uncertainty of the future forecast.

    Examples
    --------
    # Supply Chain: Simulating future demand (7 days ahead) for inventory planning
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(historical_demand, trend='add', seasonal='add').fit()
    demand_sim = CreateSLURPDistributionFromExponentialSmoothing(
        exponential_smoothing_model=model,
        forecast_steps=7,
        lower_bound=0,
        variable_name='Future Demand (7-day)',
        title_for_plot="Weekly Demand Uncertainty Projection"
    )

    # Healthcare: Projecting patient admissions for the next billing cycle
    admissions_model = sm.tsa.ETSModel(historical_admissions, error='add').fit()
    admissions_sim = CreateSLURPDistributionFromExponentialSmoothing(
        exponential_smoothing_model=admissions_model,
        forecast_steps=30,
        lower_bound=0,
        number_of_trials=5000,
        fill_color="#b0170c"
    )
    """
    # Lazy load uncommon packages
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.exponential_smoothing.ets import ETSResults
    from .pymetalog import pymetalog
    # Create an instance of the pymetalog class
    pm = pymetalog()

    # # Ensure that the exponential_smoothing_model is a statsmodels exponential smoothing model
    # if not isinstance(exponential_smoothing_model, ExponentialSmoothing):
    #     raise ValueError("exponential_smoothing_model must be a fitted statsmodels ExponentialSmoothing model.")
    
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
    z_score = 1.96 if prediction_interval == 0.95 else np.abs(stats.norm.ppf(1 - alpha/2))
    
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
