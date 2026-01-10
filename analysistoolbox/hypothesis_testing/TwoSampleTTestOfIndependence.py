# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import textwrap

# Declare function
def TwoSampleTTestOfIndependence(dataframe,
                                 outcome_column,
                                 grouping_column,
                                 confidence_interval=.95,
                                 alternative_hypothesis="two-sided",
                                 homogeneity_of_variance=True,
                                 null_handling='omit',
                                 plot_sample_distributions=True,
                                 color_palette='Set2',
                                 title_for_plot="T-Test of Independence",
                                 subtitle_for_plot="Shows the distribution of the sample means for each group.",
                                 caption_for_plot=None,
                                 data_source_for_plot=None,
                                 show_y_axis=False,
                                 title_y_indent=1.1,
                                 subtitle_y_indent=1.05,
                                 caption_y_indent=-0.15,
                                 figure_size=(8, 6)):
    """
    Perform a two-sample independent t-test to compare means between two groups.

    This function conducts a two-sample independent t-test to determine if there is a
    statistically significant difference between the means of two unrelated groups. It
    is a fundamental inferential statistic used to compare the central tendencies of
    distinct populations or experimental groups.

    A two-sample t-test is essential for:
      * Comparing experimental results between a treatment group and a control group
      * Analyzing performance differences between two distinct demographic segments
      * Testing if two manufacturing processes produce parts with different average sizes
      * Evaluating the impact of two different marketing strategies on customer spend
      * Assessing if gender, age, or location groups exhibit different average behaviors
      * Validating whether a training program significantly improved scores vs. a baseline
      * Comparing product satisfaction ratings between two different user categories

    The function calculates the t-statistic and p-value using `scipy.stats.ttest_ind`.
    It supports assumptions of both equal and unequal variances (Welch's t-test),
    provides optional visualization of the simulated sample mean distributions for both
    groups via kernel density estimates (KDE), and automatically prints a statistical
    interpretation of the results based on the provided confidence level.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the data to be analyzed.
    outcome_column
        Name of the column in the DataFrame containing the continuous dependent variable.
    grouping_column
        Name of the column in the DataFrame containing the categorical independent variable
        (must have exactly two unique groups).
    confidence_interval
        The confidence level used to determine statistical significance (e.g., 0.95
        for a 5% significance level). Defaults to 0.95.
    alternative_hypothesis
        Defines the alternative hypothesis. Must be one of 'two-sided' (means are not
        equal), 'less' (mean of first group is less than second), or 'greater'
        (mean of first group is greater than second). Defaults to 'two-sided'.
    homogeneity_of_variance
        If True, assumes equality of variance between groups (Student's t-test). If
        False, performs Welch's t-test for unequal variances. Defaults to True.
    null_handling
        Method for handling missing values (NaN) in the data. Options include
        'omit' (ignore NaNs), 'raise' (throw error), or 'propagate' (return NaN).
        Defaults to 'omit'.
    plot_sample_distributions
        If True, displays a kernel density estimate of the simulated sample means for
        both groups to visualize the comparison. Defaults to True.
    color_palette
        Seaborn color palette name used for the lines and fills in the plot.
        Defaults to 'Set2'.
    title_for_plot
        Main title text to display at the top of the plot. Defaults to
        'T-Test of Independence'.
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
    # Compare average sales between two different regions
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        'sales': np.random.normal(500, 50, 40),
        'region': ['East'] * 20 + ['West'] * 20
    })
    results = TwoSampleTTestOfIndependence(
        dataframe=df,
        outcome_column='sales',
        grouping_column='region'
    )

    # Welch's t-test for groups with unequal variances and a directional hypothesis
    recovery_df = pd.DataFrame({
        'recovery_days': [7, 8, 5, 9, 6] * 10 + [12, 14, 11, 15, 13] * 10,
        'treatment': ['Medicine A'] * 50 + ['Medicine B'] * 50
    })
    results = TwoSampleTTestOfIndependence(
        dataframe=recovery_df,
        outcome_column='recovery_days',
        grouping_column='treatment',
        homogeneity_of_variance=False,
        alternative_hypothesis='less',
        title_for_plot='Efficacy of Medicine A vs. Medicine B'
    )

    # Statistical test without visualization
    results = TwoSampleTTestOfIndependence(
        dataframe=df,
        outcome_column='sales',
        grouping_column='region',
        plot_sample_distributions=False
    )

    """
    
    # Get groupings (there should only be two)
    list_groups = dataframe[grouping_column].unique()
    if len(list_groups) > 2:
        raise ValueError("T-Test can only be performed with two groups. The grouping column you selected has " + str(len(list_groups)) + " groups.")
    
    # Show pivot table summary
    table_groups = pd.pivot_table(dataframe, 
        values=[outcome_column], 
        index=[grouping_column],
        aggfunc={outcome_column: [len, min, max, np.mean, np.std]}
    )
    print(table_groups)
    print("\n")
    
    # Restructure data
    dataframe = dataframe[[
        outcome_column,
        grouping_column,
    ]]
    group1 = dataframe[dataframe[grouping_column]==list_groups[0]]
    group2 = dataframe[dataframe[grouping_column]==list_groups[1]]
    
    # Conduct two sample t-test
    ttest_results = ttest_ind(
        group1[outcome_column], 
        group2[outcome_column], 
        axis=0,
        nan_policy=null_handling, 
        alternative=alternative_hypothesis,
        equal_var=homogeneity_of_variance
    )
    
    # Get the p-value
    p_value = ttest_results.pvalue
    
    # Get the test statistic
    test_statistic = ttest_results.statistic
    
    # If plot_distribution_of_differences is True, plot distribution of differences
    if plot_sample_distributions:
        # Create placeholder for results
        data_distribution = pd.DataFrame()
        
        # Get unqiue groups
        list_of_groups = dataframe[grouping_column].unique()
        
        # Iterate through groups
        for group in list_of_groups:
            # Get the group's data
            group_data = dataframe[dataframe[grouping_column] == group][outcome_column]
            
            # Get the group's mean
            group_mean = group_data.mean()
            
            # Get the group's standard error
            group_se = group_data.sem()
            
            # Generate a normal distribution
            group_distribution = np.random.normal(group_mean, group_se, 10000)
            
            # Convert to dataframe
            df_group_distribution = pd.DataFrame(group_distribution,
                                                 columns=[outcome_column])
            
            # Add group name
            df_group_distribution[grouping_column] = group
            
            # Row bind to data_distribution
            data_distribution = pd.concat([data_distribution, df_group_distribution], axis=0).reset_index(drop=True)
            
        # Plot distribution of means
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
            
        # Generate density plot using seaborn
        sns.kdeplot(
            data=data_distribution,
            x=outcome_column,
            hue=grouping_column,
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

    # Print interpretation of results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the sample means of the groups.")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean of the " + str(list_groups[0]) + " group is " + alternative_hypothesis + " than that of the " + str(list_groups[1]) + " group to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(p_value, 3)))
    print("statistic:", str(round(test_statistic, 3)))
    
    # Return results
    return ttest_results

