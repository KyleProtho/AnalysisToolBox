# Load packages
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def OneSampleTTest(dataframe, 
                   value_column, 
                   hypothesized_value,
                   confidence_interval=.95, 
                   alternative_hypothesis="two-sided",
                   null_handling='omit',
                   plot_sample_distribution=True,
                   # Plotting arguments
                   fill_color="#999999",
                   fill_transparency=0.2,
                   title_for_plot="One Sample T-Test",
                   subtitle_for_plot="Shows the distribution of the sample mean and the hypothesized value.",
                   caption_for_plot=None,
                   data_source_for_plot=None,
                   show_y_axis=False,
                   title_y_indent=1.10,
                   subtitle_y_indent=1.05,
                   caption_y_indent=-0.15,
                   figure_size=(8, 6)):
    # Conduct one sample t-test
    ttest_results = ttest_1samp(
        dataframe[value_column], 
        popmean=hypothesized_value, 
        axis=0, 
        nan_policy=null_handling, 
        alternative=alternative_hypothesis
    )
    
    # Calculate sample size
    sample_size = len(dataframe[value_column])
    
    # Calculate degrees of freedom
    degrees_of_freedom = sample_size - 1
    
    # Calculate the sample mean
    sample_mean = dataframe[value_column].mean()
    
    # Calculate the standard error of the mean
    standard_error_of_the_mean = dataframe[value_column].sem()
    
    # Get the t-statistic
    t_statistic = ttest_results.statistic
    
    # Get the p-value
    p_value = ttest_results.pvalue
    
    # Plot the sample distribution and the hypothesized value
    if plot_sample_distribution:
        # Generate an array of sample means
        sample_means = np.random.normal(
            loc=sample_mean, 
            scale=standard_error_of_the_mean, 
            size=10000
        )
        
        # Convert array to dataframe
        sample_means = pd.DataFrame(sample_means,
                                    columns=[value_column])
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        ax = sns.kdeplot(
            data=sample_means,
            x=value_column,
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
            x=hypothesized_value,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=hypothesized_value, 
            y=plt.ylim()[1] * 0.97, 
            s='Hypothesized value: {:.2f}'.format(hypothesized_value)+' ({:.1%})'.format(p_value),
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
        print("There is a statistically detectable difference between the hypothesized mean and the sample mean.")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean is " + alternative_hypothesis + " than the hypothesized mean to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(ttest_results.pvalue, 3)))
    print("statistic:", str(round(ttest_results.statistic, 3)))
    
    # Return results
    return ttest_results


# # Test function
# # Load data
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# iris['species'] = iris['species'].astype('category')
# # Run function
# results = OneSampleTTest(
#     dataframe=iris,
#     value_column='sepal length (cm)',
#     hypothesized_value=6.0,
# ) 
