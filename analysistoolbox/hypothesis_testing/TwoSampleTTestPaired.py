# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import textwrap

# Declare function
def TwoSampleTTestPaired(dataframe,
                         outcome1_column,
                         outcome2_column,
                         confidence_interval=.95,
                         alternative_hypothesis="two-sided",
                         null_handling='omit',
                         plot_sample_distributions=True,
                         fill_color="#999999",
                         fill_transparency=0.2,
                         title_for_plot="Paired T-Test",
                         subtitle_for_plot="Shows the distribution of the mean difference across all pairs",
                         caption_for_plot=None,
                         data_source_for_plot=None,
                         show_y_axis=False,
                         title_y_indent=1.1,
                         subtitle_y_indent=1.05,
                         caption_y_indent=-0.15,
                         figure_size=(8, 6)):
    """
    Perform a paired two-sample t-test to compare means between related observations.

    This function conducts a paired (dependent) t-test to determine if there is a
    statistically significant difference between the means of two related groups. It
    is used when the same subjects are measured twice (e.g., before and after an
    intervention) or when subjects are matched in pairs.

    A paired t-test is essential for:
      * Evaluating treatment efficacy in pre-test and post-test study designs
      * Analyzing performance changes in employees before and after training
      * Testing if a new software update improved system processing speeds
      * Comparing identical twins or matched pairs on a specific metric
      * Assessing seasonal variations where the same entities are tracked over time
      * Measuring the impact of a marketing campaign on individual customer spend
      * Validating the consistency of two different measurement tools on the same subjects

    The function calculates the t-statistic and p-value using `scipy.stats.ttest_rel`.
    It provides optional visualization of the simulated distribution of the mean
    difference relative to the null hypothesis (zero difference), helping to
    visualize the magnitude and significance of the observed change.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the paired outcome data.
    outcome1_column
        Name of the column containing the first set of observations (e.g., 'Pre-test').
    outcome2_column
        Name of the column containing the second set of observations (e.g., 'Post-test').
    confidence_interval
        The confidence level used to determine statistical significance (e.g., 0.95
        for a 5% significance level). Defaults to 0.95.
    alternative_hypothesis
        Defines the alternative hypothesis. Must be one of 'two-sided' (means are
        not equal), 'greater' (mean of first is greater than second), or 'less'
        (mean of first is less than second). Defaults to 'two-sided'.
    null_handling
        Method for handling missing values (NaN) in the data. Options include
        'omit' (ignore pairs with NaNs), 'raise' (throw error), or 'propagate'
        (return NaN). Defaults to 'omit'.
    plot_sample_distributions
        If True, displays a kernel density estimate of the simulated mean difference
        distribution to visualize the comparison against zero. Defaults to True.
    fill_color
        The fill color for the distribution plot area. Defaults to "#999999".
    fill_transparency
        Transparency level (alpha) for the distribution plot fill, ranging from 0 to 1.
        Defaults to 0.2.
    title_for_plot
        Main title text to display at the top of the plot. Defaults to 'Paired T-Test'.
    subtitle_for_plot
        Subtitle text to display below the main title. Defaults to
        'Shows the distribution of the mean difference across all pairs'.
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
    scipy.stats._stats_py.TtestResult
        A named tuple containing the calculated t-statistic and the p-value.
        Values can be accessed via `result.statistic` and `result.pvalue`.

    Examples
    --------
    # Compare student test scores before and after a training seminar
    import pandas as pd
    test_df = pd.DataFrame({
        'before': [75, 82, 68, 91, 77] * 10,
        'after': [80, 85, 72, 90, 81] * 10
    })
    results = TwoSampleTTestPaired(
        dataframe=test_df,
        outcome1_column='before',
        outcome2_column='after',
        alternative_hypothesis='less'
    )

    # Analyze processing speed improvements in a system update
    latency_df = pd.DataFrame({
        'v1_ms': [120, 150, 110, 130, 140] * 10,
        'v2_ms': [115, 140, 105, 125, 135] * 10
    })
    results = TwoSampleTTestPaired(
        dataframe=latency_df,
        outcome1_column='v1_ms',
        outcome2_column='v2_ms',
        title_for_plot='Latency Comparison: V1 vs. V2',
        fill_color='green'
    )

    # Statistical test without visualization
    results = TwoSampleTTestPaired(
        dataframe=test_df,
        outcome1_column='before',
        outcome2_column='after',
        plot_sample_distributions=False
    )

    """
    
    # Show pivot table summary
    table_groups = dataframe.groupby(
        [True]*len(dataframe)
    ).agg(
        {
            outcome1_column: [min, max, np.mean, np.std],
            outcome2_column: [min, max, np.mean, np.std]
        }
    )
    print(table_groups)
    print("\n")
    
    # Conduct two sample t-test
    ttest_results = ttest_rel(
        dataframe[outcome1_column], 
        dataframe[outcome2_column], 
        axis=0,
        nan_policy=null_handling, 
        alternative=alternative_hypothesis
    )
    
    # Get the p-value
    p_value = ttest_results.pvalue
    
    # Get the test statistic
    test_statistic = ttest_results.statistic
    
    if plot_sample_distributions:
        # Calculate the differences in each observation
        dataframe['Difference'] = dataframe[outcome1_column] - dataframe[outcome2_column]
        
        # Calculate the average difference
        sample_mean = dataframe['Difference'].mean()
        
        # Calculate the standard error of the mean
        standard_error = dataframe['Difference'].sem()
        
        # Generate normal distribution of differences
        dataframe = pd.DataFrame(
            np.random.normal(loc=sample_mean,
                            scale=standard_error,
                            size=10000),
            columns=['Difference']
        )
            
        # Plot distribution of means
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        ax = sns.kdeplot(
            data=dataframe,
            x='Difference',
            color=fill_color,
            alpha=fill_transparency,
            fill=True,
            common_norm=True
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
            x=0,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=0, 
            y=plt.ylim()[1] * 0.97, 
            s='No difference ({:.1%})'.format(p_value),
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

    # Print interpretation of results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the sample means of " + outcome1_column + " and " + outcome2_column + ".")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean of " + outcome1_column + " is " + alternative_hypothesis + " than that of " + outcome2_column + " to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(p_value, 3)))
    print("statistic:", str(round(test_statistic, 3)))
    
    # Return results
    return ttest_results

