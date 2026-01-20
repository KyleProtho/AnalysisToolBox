# Load packages
import os
import pandas as pd
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def CreateSLURPDistributionFromLogisticRegression(logistic_regression_model, 
                            # Simulation parameters
                            list_of_prediction_values,
                            number_of_trials=10000,
                            prediction_interval=0.95,
                            lower_bound=0,
                            upper_bound=1,
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
    Generate a SLURP distribution for predicted probabilities using a fitted Logistic Regression model.

    This function creates a SLURP (Simulation of Linear Uncertainty and Range Projections)
    distribution by extracting confidence intervals for predicted probabilities from a
    logistic regression model and fitting a flexible Metalog distribution to them. Unlike
    linear regression SLURPs which model outcome values, this function models the 
    uncertainty in the *probability* of a binary outcome (e.g., success/failure). This 
    is essential for propagating classification uncertainty into larger Monte Carlo 
    simulations and risk models.

    SLURP distributions from logistic regression are essential for:
      * Healthcare: Simulating the probability of disease presence based on clinical diagnostic markers.
      * Finance: Estimating the likelihood of loan default based on credit history and income levels.
      * Intelligence Analysis: Modeling the probability of a specific threat or event based on signal data.
      * Marketing: Simulating customer churn probability for a cohort based on engagement metrics.
      * Epidemiology: Estimating the individual probability of infection based on exposure risk factors.
      * Cybersecurity: Modeling the likelihood of a system breach based on detected vulnerabilities.
      * Legal Analysis: Estimating the probability of a favorable verdict based on case characteristics.
      * Environmental Science: Assessing the likelihood of extreme weather events or wildfires.

    The function uses `statsmodels` to calculate the confidence interval for the predicted 
    probability at the input `list_of_prediction_values`, then fits a Metalog distribution 
    to the interval bounds.

    Parameters
    ----------
    logistic_regression_model : statsmodels.discrete.discrete_model.BinaryResultsWrapper
        A fitted logit or probit model from the statsmodels library.
    list_of_prediction_values : list
        A list of values for each predictor in the model. If the model includes a constant, 
        it will be automatically handled.
    number_of_trials : int, optional
        The number of stochastic trials to generate from the probability uncertainty. 
        Defaults to 10000.
    prediction_interval : float, optional
        The confidence level for the predicted probability interval (between 0 and 1). 
        Defaults to 0.95.
    lower_bound : float, optional
        A logical lower limit for the probability (clamped to 0). Defaults to 0.
    upper_bound : float, optional
        A logical upper limit for the probability (clamped to 1). Defaults to 1.
    learning_rate : float, optional
        The step length for the Metalog fitting algorithm. Defaults to 0.01.
    term_maximum : int, optional
        The maximum number of terms in the Metalog expansion (up to 9). Higher terms 
        increase flexibility. Defaults to 3.
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
        Whether to display a histogram of the generated probability samples. Defaults to True.
    figure_size : tuple, optional
        The size of the plot figure in inches (width, height). Defaults to (8, 6).
    fill_color : str, optional
        The hex color code for the histogram bars. Defaults to "#999999".
    fill_transparency : float, optional
        The transparency level (0-1) for the histogram plot. Defaults to 0.6.
    show_mean : bool, optional
        Whether to display the mean probability as a vertical dashed line. Defaults to True.
    show_median : bool, optional
        Whether to display the median probability as a vertical dotted line. Defaults to True.
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
        The generated probability samples representing the uncertainty of the model's prediction.

    Examples
    --------
    # Finance: Simulating the probability of loan default for an applicant
    import statsmodels.api as sm
    logit_model = sm.Logit(y, X).fit()
    default_prob_sim = CreateSLURPDistributionFromLogisticRegression(
        logistic_regression_model=logit_model,
        list_of_prediction_values=[720, 50000],  # Credit score and Income
        title_for_plot="Uncertainty in Default Probability"
    )

    # Intelligence: Assessing signal confidence regarding a potential event
    threat_prob_sim = CreateSLURPDistributionFromLogisticRegression(
        logistic_regression_model=threat_model,
        list_of_prediction_values=[0.85, 12, 1],  # Reliability, Latency, Target_Type
        prediction_interval=0.90,
        fill_color="#c0392b"
    )
    """
    # Lazy load uncommon packages
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import BinaryResultsWrapper
    from .pymetalog import pymetalog
    # Create an instance of the pymetalog class
    pm = pymetalog()

    # Ensure that the logistic_regression_model is a statsmodels logistic regression model
    if not isinstance(logistic_regression_model, BinaryResultsWrapper):
        raise ValueError("logistic_regression_model must be a statsmodels logistic regression model.")
    
    # Get the list of predictors from the model
    list_of_predictors = logistic_regression_model.model.exog_names
    
    # If 'const' is in the list of predictors, add 1 to the list_of_prediction_values
    if 'const' in list_of_predictors:
        list_of_prediction_values = [1] + list_of_prediction_values
    
    # Get the outcome variable from the model
    outcome_variable = logistic_regression_model.model.endog_names[0]
    
    # Ensure that the length of the list_of_predictors and list_of_prediction_values are equal
    if len(list_of_predictors) != len(list_of_prediction_values):
        raise ValueError("The length of list_of_predictors and list_of_prediction_values must be equal.")
    
    # Ensure that prediction_interval is between 0 and 1
    if prediction_interval <= 0 or prediction_interval >= 1:
        raise ValueError("prediction_interval must be between 0 and 1.")
    
    # Set default bounds for probabilities if not specified
    if lower_bound is None:
        lower_bound = 0.0
    if upper_bound is None:
        upper_bound = 1.0
    
    # Get the confidence interval for predicted probabilities
    # Note: For logistic regression, get_prediction returns confidence intervals for probabilities
    pred = logistic_regression_model.get_prediction(list_of_prediction_values)
    pred_interval = pred.summary_frame(alpha=1-prediction_interval)
    
    # Debug: Print available columns to understand the structure
    # print("Available columns in pred_interval:", pred_interval.columns.tolist())
    # print("pred_interval shape:", pred_interval.shape)
    # print("pred_interval:\n", pred_interval)
    
    # Get the list of mean, lower bound, and upper bound values from pred_interval
    # For logistic regression, these are confidence intervals for predicted probabilities
    list_of_values = []
    
    # Check which columns are available and use the appropriate ones
    if 'mean' in pred_interval.columns:
        # Standard statsmodels format
        for value in ['mean', 'ci_lower', 'ci_upper']:
            list_of_values.append(pred_interval[value].values[0])
    elif 'predicted' in pred_interval.columns:
        # Alternative format
        for value in ['predicted', 'ci_lower', 'ci_upper']:
            list_of_values.append(pred_interval[value].values[0])
    else:
        # Fallback: use the first three columns
        available_cols = pred_interval.columns.tolist()
        if len(available_cols) >= 3:
            for i in range(3):
                list_of_values.append(pred_interval.iloc[0, i])
        else:
            raise ValueError(f"Unexpected prediction interval format. Available columns: {available_cols}")
    
    # print("Values before adjustment: ", list_of_values)
    
    # If the minimum and maximum values are outside the bounds, set them to the bounds
    if lower_bound is not None and list_of_values[1] < lower_bound:
        if lower_bound < 0:
            list_of_values[1] = float(lower_bound) * 0.99999
        else:
            list_of_values[1] = float(lower_bound) * 1.00001
    if upper_bound is not None and list_of_values[2] > upper_bound:
        if upper_bound < 0:
            list_of_values[2] = float(upper_bound) * 1.00001
        else:
            list_of_values[2] = float(upper_bound) * 0.99999
        
    # If the mean is outside the bounds, slightly above/below the reference bound
    if lower_bound is not None and list_of_values[0] < lower_bound:
        if lower_bound < 0:
            list_of_values[0] = list_of_values[1] * 0.99999
        else:
            list_of_values[0] = list_of_values[1] * 1.00001
    if upper_bound is not None and list_of_values[0] > upper_bound:
        if upper_bound < 0:
            list_of_values[0] = list_of_values[2] * 1.00001
        else:
            list_of_values[0] = list_of_values[2] * 0.99999
    # print("Values after adjustment: ", list_of_values)
        
    # Set the list of percentiles
    list_of_percentiles = [
        (1 - prediction_interval) / 2,  # Lower bound
        0.5,  # Middle
        1 - (1 - prediction_interval) / 2  # Upper bound
    ]
    # print("Percentiles: ", list_of_percentiles)
    
    # Use the confidence interval to simulate the probability distribution
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
    
    # Convert the array to a dataframe
    metalog_df = pd.DataFrame(arr_metalog, columns=[f"{outcome_variable}_probability"])
    
    # Plot the metalog distribution
    if show_distribution_plot:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=metalog_df,
            x=f"{outcome_variable}_probability",
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
            mean = metalog_df[f"{outcome_variable}_probability"].mean()
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
                s='Mean: {:.3f}'.format(mean),
                horizontalalignment='center',
                fontname="Arial",
                fontsize=9,
                color="#262626",
                alpha=0.75
            )
        
        # Show the median if requested
        if show_median:
            # Calculate the median
            median = metalog_df[f"{outcome_variable}_probability"].median()
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
                s='Median: {:.3f}'.format(median),
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
