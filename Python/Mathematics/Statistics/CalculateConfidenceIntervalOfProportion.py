# Load packages
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")


# Declare function
def CalculateConfidenceIntervalOfProportion(sample_proportion, 
                                            sample_size, 
                                            confidence_level=.95,
                                            plot_confidence_interval=True,
                                            name_of_variable='Proportion',
                                            # Histogram formatting arguments
                                            color_palette="Set2",
                                            fill_transparency=0.6,
                                            show_mean=True,
                                            # Text formatting arguments
                                            title_for_plot="Distribution of Sample Proportion",
                                            subtitle_for_plot=None,
                                            caption_for_plot=None,
                                            data_source_for_plot=None,
                                            show_y_axis=False,
                                            title_y_indent=1.1,
                                            subtitle_y_indent=1.05,
                                            caption_y_indent=-0.15,
                                            # Plot formatting arguments
                                            figure_size=(8, 6)):
    # Calculate the standard error
    standard_error = math.sqrt((sample_proportion * (1 - sample_proportion)) / sample_size)
    
    # Calculate the margin of error
    margin_of_error = stats.norm.ppf(1 - ((1 - confidence_level) / 2)) * standard_error
    
    # Calculate the confidence interval
    lower_bound = sample_proportion - margin_of_error
    upper_bound = sample_proportion + margin_of_error
    
    # Generate plot if user requests it
    if plot_confidence_interval:
        # Generate normal distribution of proportions
        dataframe = pd.DataFrame(np.random.normal(loc=sample_proportion,
                                                  scale = standard_error,
                                                  size = 10000),
                                    columns=[name_of_variable])
        
        # Add flag for whether the proportion is within the confidence interval
        dataframe['Within confidence interval'] = np.where((dataframe[name_of_variable] >= lower_bound) & (dataframe[name_of_variable] <= upper_bound), True, False)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=dataframe,
            x=name_of_variable,
            hue='Within confidence interval',
            palette=color_palette,
            alpha=fill_transparency,
        )
        
        # Remove the legend
        ax.legend_.remove()
        
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
        
        # Show the sample proportion as a vertical line with a label
        ax.axvline(
            x=sample_proportion,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=sample_proportion, 
            y=plt.ylim()[1] * 0.97, 
            s='Sample proportion: {:.1%}'.format(sample_proportion),
            horizontalalignment='center',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
        
        # Show the lower bound of the confidence interval as a vertical line with a label
        ax.axvline(
            x=lower_bound,
            ymax=0.22-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=lower_bound, 
            y=plt.ylim()[1] * 0.22, 
            s='Lower bound: {:.1%}'.format(lower_bound),
            horizontalalignment='right',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
        
        # Show the upper bound of the confidence interval as a vertical line with a label
        ax.axvline(
            x=upper_bound,
            ymax=0.22-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=upper_bound, 
            y=plt.ylim()[1] * 0.22, 
            s='Upper bound: {:.1%}'.format(upper_bound),
            horizontalalignment='left',
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
        
    # Print the results
    print("Sample proportion: {:.1%}".format(sample_proportion))
    print("Standard error: {:.4f}".format(standard_error))
    print("Margin of error: {:.4f}".format(margin_of_error))
    print("Confidence interval: {:.1%} to {:.1%}".format(lower_bound, upper_bound))


# # Test the function
# CalculateConfidenceIntervalOfProportion(sample_proportion=.37,
#                                         sample_size=411,
#                                         confidence_level=.95,
#                                         subtitle_for_plot="Shows the sample proportion and its 95% confidence interval")

