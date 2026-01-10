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
    Perform a two-sample independent t-test using summary statistics.

    This function conducts a two-sample independent t-test based on provided summary
    statistics (means, standard deviations, and sample sizes) for two groups. It
    determines whether there is a statistically significant difference between the
    population means of the two independent groups.

    A two-sample t-test from statistics is essential for:
      * Comparing A/B test results when only aggregate data is available
      * Benchmarking performance across different regions or departments
      * Validating experimental outcomes from published research papers
      * Testing the efficacy of a treatment group relative to a control group
      * Analyzing variations in service delivery times between two providers
      * Evaluating differences in customer lifetime value across segments
      * Assessing the impact of environmental factors on outcome distributions

    The function uses `scipy.stats.ttest_ind_from_stats` to calculate the t-statistic
    and p-value. It supports assumptions of both equal and unequal variances
    (Welch's t-test) and provides a visualization of simulated sample mean
    distributions for both groups, allowing for an intuitive comparison of their
    central tendencies and overlaps.

    Parameters
    ----------
    group1_mean
        The mean value calculated from the first sample.
    group1_sd
        The standard deviation calculated from the first sample.
    group1_sample_size
        The number of observations in the first sample.
    group2_mean
        The mean value calculated from the second sample.
    group2_sd
        The standard deviation calculated from the second sample.
    group2_sample_size
        The number of observations in the second sample.
    homogeneity_of_variance
        If True, assumes equality of variance between the two groups (Student's
        t-test). If False, performs Welch's t-test for unequal variances.
        Defaults to True.
    confidence_interval
        The confidence level used to determine statistical significance (e.g., 0.95
        for a 5% signficance level). Defaults to 0.95.
    alternative_hypothesis
        Defines the alternative hypothesis for the test. Must be one of 'two-sided',
        'less', or 'greater'. Defaults to 'two-sided'.
    plot_sample_distributions
        If True, displays a kernel density estimate of simulated sample means for
        both groups to visualize the comparison. Defaults to True.
    value_name
        Descriptive name for the value being measured, used as the x-axis label in
        the visualization. Defaults to 'Value'.
    color_palette
        Seaborn color palette name or list of colors used for the group distributions.
        Defaults to 'Set2'.
    title_for_plot
        Main title text to display at the top of the plot. Defaults to
        'T-Test of Two Means'.
    subtitle_for_plot
        Subtitle text to display below the main title. Defaults to
        'Shows the distribution of the sample means for each group.'.
    caption_for_plot
        Caption text displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot
        Optional text identifying the data source, displayed in the caption area.
        Defaults to None.
    show_y_axis
        If True, displays the y-axis (density scale) on the distribution plot.
        Defaults to False.
    title_y_indent
        Vertical position for the main title relative to the axes. Defaults to 1.1.
    subtitle_y_indent
        Vertical position for the subtitle relative to the axes. Defaults to 1.05.
    caption_y_indent
        Vertical position for the caption relative to the axes. Defaults to -0.15.
    figure_size
        Tuple specifying the (width, height) of the figure in inches. Defaults to (8, 6).

    Returns
    -------
    scipy.stats._stats_py.Ttest_indResult
        A named tuple containing the calculated t-statistic and the p-value.
        Values can be accessed via `result.statistic` and `result.pvalue`.

    Examples
    --------
    # Compare average spend between two marketing variants
    result = TTestOfTwoMeansFromStats(
        group1_mean=45.2, group1_sd=12.5, group1_sample_size=150,
        group2_mean=48.7, group2_sd=13.2, group2_sample_size=150
    )
    print(f"P-value: {result.pvalue:.4f}")

    # Welch's t-test for groups with unequal variances and sizes
    result = TTestOfTwoMeansFromStats(
        group1_mean=105.0, group1_sd=5.0, group1_sample_size=20,
        group2_mean=112.0, group2_sd=15.0, group2_sample_size=100,
        homogeneity_of_variance=False,
        alternative_hypothesis='less'
    )

    # Test with custom visualization styling
    result = TTestOfTwoMeansFromStats(
        group1_mean=75.0, group1_sd=8.0, group1_sample_size=50,
        group2_mean=80.0, group2_sd=7.5, group2_sample_size=55,
        value_name='Test Scores',
        title_for_plot='Comparison of Method A vs Method B',
        color_palette=['#3498db', '#e74c3c']
    )

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

