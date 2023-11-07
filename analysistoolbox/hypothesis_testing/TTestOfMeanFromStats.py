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
    Conducts a hypothesis test of a mean using a t-test or z-test, depending on the sample size.

    Args:
        sample_mean (float): The sample mean.
        sample_sd (float): The sample standard deviation.
        sample_size (int): The sample size.
        hypothesized_mean (float): The hypothesized mean.
        alternative_hypothesis (str, optional): The alternative hypothesis. Can be "two-sided", "less", or "greater".
            Defaults to "two-sided".
        confidence_interval (float, optional): The confidence level for the hypothesis test. Defaults to 0.95.
        plot_sample_distribution (bool, optional): Whether to plot the distribution of the sample mean and the
            hypothesized mean. Defaults to True.
        value_name (str, optional): The name of the value being tested. Defaults to "Value".
        fill_color (str, optional): The fill color for the histogram. Defaults to "#999999".
        fill_transparency (float, optional): The transparency of the fill color for the histogram. Defaults to 0.6.
        title_for_plot (str, optional): The title for the plot. Defaults to "Hypothesis Test of a Mean".
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to "Shows the distribution of the sample
            mean and the hypothesized mean.".
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis. Defaults to False.
        title_y_indent (float, optional): The y-indent for the title. Defaults to 1.10.
        subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.15.
        figure_size (tuple, optional): The size of the figure. Defaults to (8, 6).

    Returns:
        float: The test statistic.
    """
    
    # If alternative hypothesis is "two-sided", then divide and add half of complement
    if alternative_hypothesis == "two-sided":
        ppf_threshold = confidence_interval + ((1 - confidence_interval) / 2)
    else:
        ppf_threshold = confidence_interval

    # If sample size 30 or more, calculate z-stat. Otherwise, calculate t-stat
    if sample_size < 30:
        test_threshold = stats.t.ppf(ppf_threshold)
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

