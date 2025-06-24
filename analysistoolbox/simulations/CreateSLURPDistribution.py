# Load packages
import os
import pandas as pd
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def CreateSLURPDistribution(linear_regression_model, 
                            # Simulation parameters
                            list_of_prediction_values,
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
    # Lazy load uncommon packages
    import statsmodels.api as sm
    from .pymetalog import pymetalog
    # Create an instance of the pymetalog class
    pm = pymetalog()

    # Ensure that the linear_regression_model is a statsmodels regression model
    if not isinstance(linear_regression_model, sm.regression.linear_model.RegressionResultsWrapper):
        raise ValueError("linear_regression_model must be a statsmodels regression model.")
    
    # Get the list of predictors from the model
    list_of_predictors = linear_regression_model.model.exog_names
    
    # If 'const' is in the list of predictors, add 1 to the list_of_prediction_values
    if 'const' in list_of_predictors:
        list_of_prediction_values = [1] + list_of_prediction_values
    
    # Get the outcome variable from the model
    outcome_variable = linear_regression_model.model.endog_names[0]
    
    # Ensure that the length of the list_of_predictors and list_of_prediction_values are equal
    if len(list_of_predictors) != len(list_of_prediction_values):
        raise ValueError("The length of list_of_predictors and list_of_prediction_values must be equal.")
    
    # Ensure that prediction_interval is between 0 and 1
    if prediction_interval <= 0 or prediction_interval >= 1:
        raise ValueError("prediction_interval must be between 0 and 1.")
    
    # Get the prediction interval
    pred = linear_regression_model.get_prediction(list_of_prediction_values)
    pred_interval = pred.summary_frame(alpha=1-prediction_interval)
    
    # Get the list of mean, lower bound, and upper bound values from pred_interval
    list_of_values = []
    for value in ['obs_ci_lower', 'mean', 'obs_ci_upper']:
        list_of_values.append(pred_interval[value].values[0])
    # print("Values before adjustment: ", list_of_values)
    
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
    # print("Values after adjustment: ", list_of_values)
        
    # Set the list of percentiles
    list_of_percentiles = [
        (1 - prediction_interval) / 2,  # Lower bound
        0.5,  # Middle
        1 - (1 - prediction_interval) / 2  # Upper bound
    ]
    # print("Percentiles: ", list_of_percentiles)
    
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

