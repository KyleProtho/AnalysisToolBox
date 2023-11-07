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
    Conducts a paired two-sample t-test on two columns of a pandas DataFrame.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data to be analyzed.
        outcome1_column (str): The name of the first column containing the outcome data.
        outcome2_column (str): The name of the second column containing the outcome data.
        confidence_interval (float, optional): The desired level of confidence for the test. Defaults to 0.95.
        alternative_hypothesis (str, optional): The alternative hypothesis to be tested. Can be "two-sided", "greater", or "less". Defaults to "two-sided".
        null_handling (str, optional): How to handle null values in the data. Can be "omit", "raise", or "propagate". Defaults to "omit".
        plot_sample_distributions (bool, optional): Whether to plot the distribution of the mean difference across all pairs. Defaults to True.
        fill_color (str, optional): The color to fill the distribution plot with. Defaults to "#999999".
        fill_transparency (float, optional): The transparency of the fill color. Defaults to 0.2.
        title_for_plot (str, optional): The title of the distribution plot. Defaults to "Paired T-Test".
        subtitle_for_plot (str, optional): The subtitle of the distribution plot. Defaults to "Shows the distribution of the mean difference across all pairs".
        caption_for_plot (str, optional): The caption of the distribution plot. Defaults to None.
        data_source_for_plot (str, optional): The data source of the distribution plot. Defaults to None.
        show_y_axis (bool, optional): Whether to show the y-axis on the distribution plot. Defaults to False.
        title_y_indent (float, optional): The y-indent of the title on the distribution plot. Defaults to 1.1.
        subtitle_y_indent (float, optional): The y-indent of the subtitle on the distribution plot. Defaults to 1.05.
        caption_y_indent (float, optional): The y-indent of the caption on the distribution plot. Defaults to -0.15.
        figure_size (tuple, optional): The size of the distribution plot. Defaults to (8, 6).

    Returns:
        scipy.stats.Ttest_relResult: The results of the paired two-sample t-test.
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

