# Load packages 
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import textwrap

# Declare function
def CalculateConfidenceIntervalOfMean(sample_mean,
                                      sample_sd,
                                      sample_size,
                                      confidence_interval=.95,
                                      # Plot parameters
                                      plot_sample_distribution=True,
                                      value_name="Possible Mean",
                                      fill_color="#999999",
                                      fill_transparency=0.6,
                                      title_for_plot="Confidence Interval of Mean",
                                      subtitle_for_plot="Shows the distribution of the sample mean.",
                                      caption_for_plot=None,
                                      data_source_for_plot=None,
                                      show_y_axis=False,
                                      title_y_indent=1.10,
                                      subtitle_y_indent=1.05,
                                      caption_y_indent=-0.15,
                                      figure_size=(8, 6)):
    """
    Calculate and visualize the confidence interval for a sample mean.

    This function determines the confidence interval for a population mean based on 
    sample statistics. It employs the t-distribution for small samples (n < 30) 
    and the z-distribution (normal distribution) for larger samples (n >= 30), 
    following the central limit theorem. The function calculates the standard error 
    of the mean and applies the appropriate critical value based on the specified 
    confidence level. Optionally, it generates a visual representation of the 
    sampling distribution to provide context for the estimate.

    Calculating confidence intervals is essential for:
      * Estimating population parameters from sample data with quantified uncertainty.
      * Epidemiology: Determining the mean incubation period of a pathogen in a clinical study.
      * Healthcare: Assessing the average reduction in blood pressure across a patient cohort.
      * Intelligence Analysis: Estimating the mean frequency of communication events from a monitored source.
      * Data Science: Evaluating the average processing time of a distributed system.
      * Quality Control: Monitoring the mean weight or dimensions of manufactured components.
      * Finance: Estimating the average daily return of an asset over a specific period.
      * Social Science: Analyzing the mean response score from a public opinion survey.

    Parameters
    ----------
    sample_mean : float
        The arithmetic mean observed in the sample data.
    sample_sd : float
        The standard deviation of the sample, used to estimate the population standard deviation.
    sample_size : int
        The number of observations in the sample. This determines whether a t-stat or z-stat is used.
    confidence_interval : float, optional
        The desired confidence level, expressed as a decimal between 0 and 1. Defaults to 0.95.
    plot_sample_distribution : bool, optional
        Whether to generate and display a histogram showing the inferred sampling distribution 
        of the mean. Defaults to True.
    value_name : str, optional
        A descriptive name for the variable being analyzed, used as the x-axis label in the plot. 
        Defaults to "Possible Mean".
    fill_color : str, optional
        The hex color code for the bars in the distribution plot. Defaults to "#999999".
    fill_transparency : float, optional
        The alpha transparency level for the plot's histogram bars (0 to 1). Defaults to 0.6.
    title_for_plot : str, optional
        The primary title text displayed at the top of the plot. Defaults to "Confidence Interval of Mean".
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the main title. Defaults to "Shows the distribution of the sample mean.".
    caption_for_plot : str, optional
        An optional descriptive caption or note displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text indicating the source of the data, appended to the caption. Defaults to None.
    show_y_axis : bool, optional
        Whether to display the frequency/density scale on the y-axis. Defaults to False.
    title_y_indent : float, optional
        Vertical position offset for the plot title. Defaults to 1.10.
    subtitle_y_indent : float, optional
        Vertical position offset for the plot subtitle. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical position offset for the plot caption. Defaults to -0.15.
    figure_size : tuple, optional
        The dimensions of the output figure as a (width, height) tuple in inches. Defaults to (8, 6).

    Returns
    -------
    None
        The function prints the calculated confidence interval to the console and displays a 
        matplotlib plot if `plot_sample_distribution` is True.

    Examples
    --------
    # Epidemiology: Estimating mean recovery time for a viral infection
    CalculateConfidenceIntervalOfMean(
        sample_mean=14.2,
        sample_sd=3.5,
        sample_size=45,
        confidence_interval=0.99,
        value_name="Days to Recovery",
        title_for_plot="99% CI for Mean Recovery Time"
    )

    # Healthcare: Analyzing patient wait times in an emergency department
    CalculateConfidenceIntervalOfMean(
        sample_mean=120,
        sample_sd=45,
        sample_size=150,
        value_name="Wait Time (Minutes)",
        subtitle_for_plot="Analysis of Q3 Patient Flow Data",
        fill_color="#3498db"
    )

    # Intelligence Analysis: Estimating mean interval between signal detections
    CalculateConfidenceIntervalOfMean(
        sample_mean=0.85,
        sample_sd=0.12,
        sample_size=25,
        confidence_interval=0.95,
        value_name="Signal Interval (Seconds)",
        title_for_plot="Signal Pulse Frequency Analysis",
        caption_for_plot="Based on intercepted transmission data from sector 7."
    )
    """
    # If sample size 30 or more, calculate z-stat. Otherwise, calculate t-stat
    if sample_size < 30:
        test_threshold = stats.t.ppf(confidence_interval,
                                     df=sample_size-1)
    else:
        test_threshold = stats.norm.ppf(confidence_interval)
    
    # Calculate standard error of sample
    standard_error = sample_sd / sqrt(sample_size)
    
    # Show the confidence interval of the sample mean
    print("The sample mean is", str(round(sample_mean, 3)), "with a", str(round(confidence_interval*100, 1))+"% confidence interval of", str(round(sample_mean - test_threshold*standard_error, 3)), "to", str(round(sample_mean + test_threshold*standard_error, 3)))
    
    # Generate plot of distribution of sample mean and hypothesized mean
    if plot_sample_distribution:
        # Generate an array of sample means
        sample_means = np.random.normal(
            loc=sample_mean, 
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
            x=sample_mean - test_threshold*standard_error,
            color="#262626",
            linestyle="--",
            linewidth=1
        )
        
        # Add data label for the sample proportion at the lower bound of the confidence interval
        ax.text(
            x=sample_mean - test_threshold*standard_error,
            y=0.9*ax.get_ylim()[1],
            s="Lower bound: " + str(round(sample_mean - test_threshold*standard_error, 3)) + "  ",
            fontsize=8,
            color="#666666",
            ha="right",
            va="bottom"
        )
        
        # Add a vertical line for the sample proportion at the upper bound of the confidence interval
        ax.axvline(
            x=sample_mean + test_threshold*standard_error,
            color="#262626",
            linestyle="--",
            linewidth=1
        )
        
        # Add data label for the sample proportion at the upper bound of the confidence interval
        ax.text(
            x=sample_mean + test_threshold*standard_error,
            y=0.9*ax.get_ylim()[1],
            s="  Upper bound: " + str(round(sample_mean + test_threshold*standard_error, 3)),
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

