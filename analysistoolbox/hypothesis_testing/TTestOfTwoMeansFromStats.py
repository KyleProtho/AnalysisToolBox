# Load packages
from scipy.stats import ttest_ind_from_stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def TTestOfTwoMeansFromStats(group1_mean,
                             group1_sd,
                             group1_sample_size,
                             group2_mean,
                             group2_sd,
                             group2_sample_size,
                             homogeneity_of_variance=True,
                             confidence_interval=0.95,
                             alternative_hypothesis="two-sided",
                             plot_sample_distributions=True,
                             value_name="Value",
                             color_palette='Set2',
                             title_for_plot="T-Test of Two Means",
                             subtitle_for_plot="Shows the distribution of the sample means for each group.",
                             caption_for_plot=None,
                             data_source_for_plot=None,
                             show_y_axis=False,
                             title_y_indent=1.1,
                             subtitle_y_indent=1.05,
                             caption_y_indent=-0.15,
                             figure_size=(8, 6)):
    """
    Conducts a two-sample t-test of the null hypothesis that the means of two independent samples are equal.

    Args:
        group1_mean (float): The mean of the first sample.
        group1_sd (float): The standard deviation of the first sample.
        group1_sample_size (int): The sample size of the first sample.
        group2_mean (float): The mean of the second sample.
        group2_sd (float): The standard deviation of the second sample.
        group2_sample_size (int): The sample size of the second sample.
        homogeneity_of_variance (bool, optional): Whether to assume equal variances for the two samples. Defaults to True.
        confidence_interval (float, optional): The confidence level for the test. Defaults to 0.95.
        alternative_hypothesis (str, optional): The alternative hypothesis for the test. Can be "two-sided", "less", or "greater". Defaults to "two-sided".
        plot_sample_distributions (bool, optional): Whether to plot the distribution of the sample means for each group. Defaults to True.
        value_name (str, optional): The name of the value being measured. Defaults to "Value".
        color_palette (str or list, optional): The color palette to use for the plot. Can be a string with the name of a seaborn palette or a list of colors. Defaults to 'Set2'.
        title_for_plot (str, optional): The title of the plot. Defaults to "T-Test of Two Means".
        subtitle_for_plot (str, optional): The subtitle of the plot. Defaults to "Shows the distribution of the sample means for each group.".
        caption_for_plot (str, optional): The caption of the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source of the plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis of the plot. Defaults to False.
        title_y_indent (float, optional): The y-coordinate of the title of the plot. Defaults to 1.1.
        subtitle_y_indent (float, optional): The y-coordinate of the subtitle of the plot. Defaults to 1.05.
        caption_y_indent (float, optional): The y-coordinate of the caption of the plot. Defaults to -0.15.
        figure_size (tuple, optional): The size of the plot. Defaults to (8, 6).

    Returns:
        namedtuple: A named tuple containing the test statistic and the p-value of the test.
    """
    
    # Conduct t-test
    ttest_results = ttest_ind_from_stats(
        mean1=group1_mean, 
        std1=group1_sd, 
        nobs1=group1_sample_size,
        mean2=group2_mean, 
        std2=group2_sd, 
        nobs2=group2_sample_size,
        equal_var=homogeneity_of_variance,
        alternative=alternative_hypothesis
    )

    # Interpret test results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the sample means of Group 1 and Group 2.")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean of Group 1 is " + alternative_hypothesis + " than that of Group 2 to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(ttest_results.pvalue, 3)))
    print("statistic:", str(round(ttest_results.statistic, 3)))
    
    # If plot_distribution_of_differences is True, plot distribution of differences
    if plot_sample_distributions:
        # Create a normal distribution for Group 1
        group1_distribution = np.random.normal(
            loc=group1_mean,
            scale=group1_sd / np.sqrt(group1_sample_size),  # Standard error
            size=10000
        )
        
        # Convert to dataframe
        group1_distribution = pd.DataFrame(group1_distribution,
                                           columns=[value_name])
        
        # Create a normal distribution for Group 2
        group2_distribution = np.random.normal(
            loc=group2_mean,
            scale=group2_sd / np.sqrt(group2_sample_size),  # Standard error
            size=10000
        )
        
        # Convert to dataframe
        group2_distribution = pd.DataFrame(group2_distribution,
                                           columns=[value_name])
        
        # Add group label
        group1_distribution["Group"] = "Group 1"
        group2_distribution["Group"] = "Group 2"
        
        # Row bind groups together
        data_distribution = pd.concat([group1_distribution, group2_distribution], axis=0).reset_index(drop=True)
        
        # Plot distribution of means
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
            
        # Generate density plot using seaborn
        sns.kdeplot(
            data=data_distribution,
            x=value_name,
            hue="Group",
            fill=True,
            common_norm=False,
            alpha=.4,
            palette=color_palette,
            linewidth=1
        )
        
        # Remove top, left, and right spines. Set bottom spine to dark gray.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color("#262626")
        
        # Remove the y-axis, and adjust the indent of the plot titles
        if show_y_axis == False:
            ax.axes.get_yaxis().set_visible(False)
            x_indent = 0.015
        else:
            x_indent = -0.005
        
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
        
    # Reutrn results
    return ttest_results

