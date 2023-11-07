# Load packages
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import seaborn as sns
import textwrap

# Declare function
def OneWayANOVA(dataframe,
                outcome_column,
                grouping_column,
                signficance_level=0.05,
                homogeneity_of_variance=True,
                use_welch_correction=False,
                plot_sample_distributions=True,
                color_palette='Set2',
                title_for_plot="One-Way ANOVA",
                subtitle_for_plot="Shows the distribution of the sample means for each group.",
                caption_for_plot=None,
                data_source_for_plot=None,
                show_y_axis=False,
                title_y_indent=1.1,
                subtitle_y_indent=1.05,
                caption_y_indent=-0.15,
                figure_size=(8, 6)):
    """
    Conducts a one-way ANOVA on a given dataframe, with options for post-hoc testing and visualization.

    Args:
        dataframe: A pandas dataframe containing the data to be analyzed.
        outcome_column: A string representing the name of the column containing the outcome variable.
        grouping_column: A string representing the name of the column containing the grouping variable.
        signficance_level: A float representing the desired significance level (default: 0.05).
        homogeneity_of_variance: A boolean indicating whether to assume homogeneity of variance (default: True).
        use_welch_correction: A boolean indicating whether to use Welch's correction for unequal variances (default: False).
        plot_sample_distributions: A boolean indicating whether to plot the distribution of sample means for each group (default: True).
        color_palette: A string representing the name of the color palette to use for the plot (default: 'Set2').
        title_for_plot: A string representing the title of the plot (default: 'One-Way ANOVA').
        subtitle_for_plot: A string representing the subtitle of the plot (default: 'Shows the distribution of the sample means for each group.').
        caption_for_plot: A string representing the caption of the plot (default: None).
        data_source_for_plot: A string representing the data source of the plot (default: None).
        show_y_axis: A boolean indicating whether to show the y-axis on the plot (default: False).
        title_y_indent: A float representing the y-axis position of the title (default: 1.1).
        subtitle_y_indent: A float representing the y-axis position of the subtitle (default: 1.05).
        caption_y_indent: A float representing the y-axis position of the caption (default: -0.15).
        figure_size: A tuple representing the size of the plot (default: (8, 6)).

    Returns:
        A string representing the results of Tukey's test.
    """
    
    # Show pivot table summary
    table_groups = pd.pivot_table(dataframe, 
        values=[outcome_column], 
        index=[grouping_column],
        aggfunc={outcome_column: [len, min, max, np.mean, np.std, np.var]}
    )
    print(table_groups)
    print("\n")

    # Set variance argument
    if homogeneity_of_variance:
        variance_arg = "equal"
    else:
        variance_arg = "unequal"

    # Conduct one-way ANOVA
    test_results = anova_oneway(
        data=dataframe[outcome_column], 
        groups=dataframe[grouping_column], 
        use_var=variance_arg, 
        welch_correction=use_welch_correction
    )

    # Print interpretation of results
    if test_results.pvalue <= signficance_level:
        print("There is a statistically detectable difference between the sample means of at least 2 groups.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(test_results.pvalue, 3)))
    print("degrees of freedom between groups:", str(int(test_results.df_num)))
    print("degrees of freedom across all groups:", str(int(test_results.df_denom)))
    print("f-statistic:", str(round(test_results.statistic, 3)))
    print("\n")
    
    # Perform Tukey's test
    tukey = pairwise_tukeyhsd(
        endog=dataframe[outcome_column],
        groups=dataframe[grouping_column],
        alpha=signficance_level
    )
    results = tukey.summary()
    
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

    # Return results
    return results

