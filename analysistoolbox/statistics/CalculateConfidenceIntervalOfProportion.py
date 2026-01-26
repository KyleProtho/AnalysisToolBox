# Load packages 
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import textwrap

# Declare function
def CalculateConfidenceIntervalOfProportion(sample_proportion,
                                            sample_size,
                                            confidence_interval=.95,
                                            # Plot parameters
                                            plot_sample_distribution=True,
                                            value_name="Possible Proportion",
                                            fill_color="#999999",
                                            fill_transparency=0.6,
                                            title_for_plot="Confidence Interval of Proportion",
                                            subtitle_for_plot="Shows the distribution of the sample proportion.",
                                            caption_for_plot=None,
                                            data_source_for_plot=None,
                                            show_y_axis=False,
                                            title_y_indent=1.10,
                                            subtitle_y_indent=1.05,
                                            caption_y_indent=-0.15,
                                            figure_size=(8, 6)):
    """
    Calculate and visualize the confidence interval for a sample proportion.

    This function determines the confidence interval for a population proportion based 
    on sample data. It employs the t-distribution for small samples (n < 30) 
    and the z-distribution (normal distribution) for larger samples (n >= 30). 
    The function calculates the standard error for proportions using the 
    formula sqrt(p*(1-p)/n) and applies the critical value corresponding to 
    the specified confidence level. A visualization of the sampling distribution 
    is optionally generated to illustrate the uncertainty of the estimate.

    Calculating confidence intervals for proportions is essential for:
      * Estimating the prevalence of a characteristic within a larger population.
      * Epidemiology: Estimating the proportion of a population infected during an outbreak.
      * Healthcare: Determining the success rate of a new clinical treatment or medication.
      * Intelligence Analysis: Assessing the probability of a specific event based on historical frequency.
      * Data Science: Evaluating click-through rates (CTR) or conversion rates in A/B testing.
      * Quality Control: Monitoring the defect rate in a manufacturing production line.
      * Marketing: Estimating the market share of a product based on survey responses.
      * Public Opinion: Calculating the margin of error for political polling results.

    Parameters
    ----------
    sample_proportion : float
        The observed proportion in the sample (e.g., 0.45 for 45%). Must be between 0 and 1.
    sample_size : int
        The total number of observations in the sample. Influences the selection of t-stat vs. z-stat.
    confidence_interval : float, optional
        The desired confidence level as a decimal (e.g., 0.95 for 95%). Defaults to 0.95.
    plot_sample_distribution : bool, optional
        Whether to generate and display a plot of the sampling distribution. Defaults to True.
    value_name : str, optional
        The label for the x-axis in the generated plot. Defaults to "Possible Proportion".
    fill_color : str, optional
        The hex color code for the histogram bars in the distribution plot. Defaults to "#999999".
    fill_transparency : float, optional
        Transparency level (alpha) for the plot bars, ranging from 0 to 1. Defaults to 0.6.
    title_for_plot : str, optional
        The primary title displayed at the top of the plot. Defaults to "Confidence Interval of Proportion".
    subtitle_for_plot : str, optional
        Text displayed below the main title. Defaults to "Shows the distribution of the sample proportion.".
    caption_for_plot : str, optional
        Optional descriptive text or notes at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    show_y_axis : bool, optional
        Whether to show the y-axis (density/frequency) in the plot. Defaults to False.
    title_y_indent : float, optional
        Vertical offset for the plot title position. Defaults to 1.10.
    subtitle_y_indent : float, optional
        Vertical offset for the plot subtitle position. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the plot caption position. Defaults to -0.15.
    figure_size : tuple, optional
        The width and height of the figure in inches as a (width, height) tuple. Defaults to (8, 6).

    Returns
    -------
    None
        The function prints the calculated interval to the console and displays a plot if requested.

    Examples
    --------
    # Epidemiology: Estimating the vaccination rate in a specific region
    CalculateConfidenceIntervalOfProportion(
        sample_proportion=0.72,
        sample_size=500,
        confidence_interval=0.95,
        value_name="Vaccination Rate",
        title_for_plot="95% CI for Community Vaccination Coverage"
    )

    # Data Science: Analyzing the conversion rate of a new website feature
    CalculateConfidenceIntervalOfProportion(
        sample_proportion=0.08,
        sample_size=1200,
        confidence_interval=0.99,
        value_name="Conversion Rate",
        fill_color="#2ecc71"
    )

    # Intelligence Analysis: Estimating the probability of a recurring signal pattern
    CalculateConfidenceIntervalOfProportion(
        sample_proportion=0.35,
        sample_size=40,
        value_name="Pattern Frequency",
        subtitle_for_plot="Probability Analysis of Signal 'Delta'",
        caption_for_plot="Based on telemetry logs from the previous 24-hour cycle."
    )
    """
    # If sample size 30 or more, calculate z-stat. Otherwise, calculate t-stat
    if sample_size < 30:
        test_threshold = stats.t.ppf(confidence_interval,
                                     df=sample_size-1)
    else:
        test_threshold = stats.norm.ppf(confidence_interval)
    
    # Calculate standard error of sample
    standard_error = sqrt((sample_proportion*(1-sample_proportion)/sample_size))
    
    # Show the confidence interval of the sample proportion
    print("The sample proportion is", str(round(sample_proportion, 3)), "with a", str(round(confidence_interval*100, 1))+"% confidence interval of", str(round(sample_proportion - test_threshold*standard_error, 3)), "to", str(round(sample_proportion + test_threshold*standard_error, 3)))
    
    # Generate plot of distribution of sample mean and hypothesized mean
    if plot_sample_distribution:
        # Generate an array of sample means
        sample_means = np.random.normal(
            loc=sample_proportion, 
            scale=standard_error, 
            size=10000
        )
        
        # Convert array to dataframe
        sample_means = pd.DataFrame(sample_means,
                                    columns=[value_name])
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        ax = sns.histplot(
            data=sample_means,
            x=value_name,
            alpha=fill_transparency,
            color=fill_color,
            bins=30
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
        
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent,
            y=title_y_indent,
            s=title_for_plot,
            # fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            # fontname="Arial",
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
        # plt.xticks(fontname='Arial')
        
        # Add a vertical line for the sample proportion at the lower bound of the confidence interval
        ax.axvline(
            x=sample_proportion - test_threshold*standard_error,
            color="#262626",
            linestyle="--",
            linewidth=1
        )
        
        # Add data label for the sample proportion at the lower bound of the confidence interval
        ax.text(
            x=sample_proportion - test_threshold*standard_error,
            y=0.9*ax.get_ylim()[1],
            s="Lower bound: " + str(round(sample_proportion - test_threshold*standard_error, 3)) + "  ",
            fontsize=8,
            color="#666666",
            ha="right",
            va="bottom"
        )
        
        # Add a vertical line for the sample proportion at the upper bound of the confidence interval
        ax.axvline(
            x=sample_proportion + test_threshold*standard_error,
            color="#262626",
            linestyle="--",
            linewidth=1
        )
        
        # Add data label for the sample proportion at the upper bound of the confidence interval
        ax.text(
            x=sample_proportion + test_threshold*standard_error,
            y=0.9*ax.get_ylim()[1],
            s="  Upper bound: " + str(round(sample_proportion + test_threshold*standard_error, 3)),
            fontsize=8,
            color="#666666",
            ha="left",
            va="bottom"
        )
        
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
