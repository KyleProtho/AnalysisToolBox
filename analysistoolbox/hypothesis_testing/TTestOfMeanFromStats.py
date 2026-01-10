# Load packages 
from scipy import stats
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def TTestOfMeanFromStats(sample_mean,
                         sample_sd,
                         sample_size,
                         hypothesized_mean,
                         alternative_hypothesis="two-sided",
                         confidence_interval=.95,
                         plot_sample_distribution=True,
                         value_name="Value",
                         fill_color="#999999",
                         fill_transparency=0.6,
                         title_for_plot="Hypothesis Test of a Mean",
                         subtitle_for_plot="Shows the distribution of the sample mean and the hypothesized mean.",
                         caption_for_plot=None,
                         data_source_for_plot=None,
                         show_y_axis=False,
                         title_y_indent=1.10,
                         subtitle_y_indent=1.05,
                         caption_y_indent=-0.15,
                         figure_size=(8, 6)):
    """
    Perform a hypothesis test of a mean using summary statistics.

    This function conducts either a one-sample t-test or a z-test based on the provided
    sample statistics (mean, standard deviation, and size) to determine if a population
    mean significantly differs from a hypothesized value. It automatically selects the
    appropriate test based on the sample size: a t-test for small samples (n < 30) or a
    z-test for larger samples (n >= 30).

    A hypothesis test of a mean from statistics is essential for:
      * Validating experimental results from summarized research papers
      * Testing quality control standards when only summary data is available
      * High-level benchmarking against industry or historical averages
      * Financial auditing and anomaly detection in aggregate data
      * Performance evaluation based on reported mean metrics
      * Scenario planning and sensitivity analysis for projected means

    The function calculates the test statistic (t or z) and the corresponding p-value.
    It supports multiple alternative hypotheses (two-sided, less, or greater) and
    provides a visualization of the simulated sample mean distribution relative to the
    hypothesized mean, making the statistical significance visually intuitive.

    Parameters
    ----------
    sample_mean
        The mean value calculated from the sample data.
    sample_sd
        The standard deviation of the sample data.
    sample_size
        The number of observations in the sample (n). If n < 30, a t-distribution is used;
        otherwise, a normal distribution is assumed.
    hypothesized_mean
        The population mean value to test against (null hypothesis value).
    alternative_hypothesis
        Defines the alternative hypothesis for the test. Must be one of 'two-sided',
        'less', or 'greater'. Defaults to 'two-sided'.
    confidence_interval
        The confidence level used to determine statistical significance (e.g., 0.95
        for a 5% signficance level). Defaults to 0.95.
    plot_sample_distribution
        If True, displays a histogram showing the distribution of simulated sample
        means relative to the hypothesized value. Defaults to True.
    value_name
        Descriptive name for the value being tested, used as the x-axis label in
        the visualization. Defaults to 'Value'.
    fill_color
        Hex color code or name for the distribution plot bars. Defaults to '#999999'.
    fill_transparency
        Transparency level (alpha) for the plot bars, ranging from 0 to 1.
        Defaults to 0.6.
    title_for_plot
        Main title text to display at the top of the plot. Defaults to
        'Hypothesis Test of a Mean'.
    subtitle_for_plot
        Subtitle text to display below the main title. Defaults to
        'Shows the distribution of the sample mean and the hypothesized mean.'.
    caption_for_plot
        Caption text displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot
        Optional text identifying the data source, displayed in the caption area.
        Defaults to None.
    show_y_axis
        If True, displays the y-axis (frequency scale) on the distribution plot.
        Defaults to False.
    title_y_indent
        Vertical position for the main title relative to the axes. Defaults to 1.10.
    subtitle_y_indent
        Vertical position for the subtitle relative to the axes. Defaults to 1.05.
    caption_y_indent
        Vertical position for the caption relative to the axes. Defaults to -0.15.
    figure_size
        Tuple specifying the (width, height) of the figure in inches. Defaults to (8, 6).

    Returns
    -------
    float
        The calculated test statistic (t-score or z-score).

    Examples
    --------
    # Test if a reported sample mean of 52 differs from a hypothesized 50
    test_stat = TTestOfMeanFromStats(
        sample_mean=52.0,
        sample_sd=5.0,
        sample_size=35,
        hypothesized_mean=50.0
    )

    # Small sample test (t-test) with a 'greater than' alternative hypothesis
    test_stat = TTestOfMeanFromStats(
        sample_mean=105.5,
        sample_sd=15.2,
        sample_size=15,
        hypothesized_mean=100.0,
        alternative_hypothesis='greater',
        value_name='IQ Scores',
        title_for_plot='Comparison of Local vs. National IQ'
    )

    # Test without visualization for quick calculation
    test_stat = TTestOfMeanFromStats(
        sample_mean=48.2,
        sample_sd=4.1,
        sample_size=100,
        hypothesized_mean=50.0,
        plot_sample_distribution=False
    )

    """
    
    # If alternative hypothesis is "two-sided", then divide and add half of complement
    if alternative_hypothesis == "two-sided":
        ppf_threshold = confidence_interval + ((1 - confidence_interval) / 2)
    else:
        ppf_threshold = confidence_interval

    # If sample size 30 or more, calculate z-stat. Otherwise, calculate t-stat
    if sample_size < 30:
        test_threshold = stats.t.ppf(ppf_threshold,
                                     df=sample_size-1)
    else:
        test_threshold = stats.norm.ppf(ppf_threshold)

    # Calculate difference between sample and hypothesized mean
    diff_in_means = hypothesized_mean - sample_mean

    # Calculate standard error of sample
    standard_error = sample_sd / sqrt(sample_size)

    # Calculate test statistic
    test_stat = diff_in_means / standard_error

    # Interpret results
    if alternative_hypothesis == "two-sided" and abs(test_stat) > test_threshold:
        print("There is a statistically detectable difference between the hypothesized mean and the sample mean.")
        if sample_size < 30:
            p_value = stats.t.cdf(abs(test_stat)*-1)*2
            print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            p_value = stats.norm.cdf(abs(test_stat)*-1)*2
            print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
    elif alternative_hypothesis == "less" and test_stat < test_threshold*-1:
        print("The hypothesized mean is", alternative_hypothesis, "than the sample mean to a statistically detectable extent.")
        if sample_size < 30:
            p_value = stats.t.cdf(abs(test_stat)*-1)
            print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            p_value = stats.norm.cdf(abs(test_stat)*-1)
            print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
    elif alternative_hypothesis == "greater" and test_stat > test_threshold:
        print("The hypothesized mean is", alternative_hypothesis, "than the sample mean to a statistically detectable extent.")
        if sample_size < 30:
            p_value = 1 - stats.t.cdf(abs(test_stat))
            print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            p_value = 1 - stats.norm.cdf(abs(test_stat))
            print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
    else:
        print("Cannot reject the null hypothesis")
        if sample_size < 30:
            if alternative_hypothesis != "greater":
                p_value = stats.t.cdf(test_stat)
                print("p-value:", str(round(p_value, 3)))
            else:
                p_value = 1 - stats.t.cdf(test_stat)
                print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            if alternative_hypothesis != "greater":
                p_value = stats.norm.cdf(test_stat)
                print("p-value:", str(round(p_value, 3)))
            else:
                p_value = 1 - stats.norm.cdf(test_stat)
                print("p-value:", str(round(p_value, 3)))
            print("statistic:", str(round(test_stat, 3)))
            
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
        
        # # Flag values if they are outside of the confidence interval
        # if alternative_hypothesis=="two-sided":
        #     sample_means['Null Rejection Zone'] = np.where(
        #         sample_means[value_name] < hypothesized_mean, True,
        #         np.where(sample_means[value_name] > hypothesized_mean, True,
        #         False
        #     ))
        # elif alternative_hypothesis=="less":
        #     sample_means['Null Rejection Zone'] = np.where(
        #         sample_means[value_name] < hypothesized_mean, True,
        #         False
        #     )
        # elif alternative_hypothesis=="greater":
        #     sample_means['Null Rejection Zone'] = np.where(
        #         sample_means[value_name] > hypothesized_mean, True,
        #         False
        #     )
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        ax = sns.histplot(
            data=sample_means,
            x=value_name,
            # hue='Null Rejection Zone',
            # palette={True: "#FF0000", False: fill_color},
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
        
        # Show the hypothesis value as a vertical line with a label
        ax.axvline(
            x=hypothesized_mean,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=hypothesized_mean, 
            y=plt.ylim()[1] * 0.97, 
            s='Hypothesized value: {:.2f}'.format(hypothesized_mean)+' ({:.1%})'.format(p_value),
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

    # Return results
    return test_stat

