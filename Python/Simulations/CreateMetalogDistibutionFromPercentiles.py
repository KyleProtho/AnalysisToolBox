# Load packages
import os
import pandas as pd
import pymetalog as pm
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def CreateMetalogDistributionFromPercentiles(list_of_values,
                                             list_of_percentiles,
                                             return_simulated_data=False,
                                             boundedness="both",
                                             term_limit=None,
                                             variable_name="Simulated Value",
                                             show_distribution_plot=True,
                                             fill_color="#999999",
                                             fill_transparency=0.6,
                                             title_for_plot=None,
                                             subtitle_for_plot=None,
                                             caption_for_plot=None,
                                             data_source_for_plot=None,
                                             show_mean=True,
                                             show_median=True,
                                             show_y_axis=False,
                                             figure_size=(8, 6),
                                             title_y_indent=1.1,
                                             subtitle_y_indent=1.05,
                                             caption_y_indent=-0.15):
    # Ensure that the list of values and list of percentiles are the same length
    if len(list_of_values) != len(list_of_percentiles):
        raise ValueError("The list of values and list of percentiles must be the same length.")
    
    # Ensure boundedness is a valid argument
    if boundedness not in ["both", "lower", "upper", "unbounded"]:
        raise ValueError("Boundedness must be one of the following: both, lower, upper, or unbounded.")
    
    # If term limit is not specified, then use the length of the scores
    if term_limit is None:
        term_limit = len(list_of_values)
        
    # Convert boundedness to the appropriate format
    if boundedness == "both":
        boundedness = "b"
    elif boundedness == "lower":
        boundedness = "sl"
    elif boundedness == "upper":
        boundedness = "su"
    else:
        boundedness = "u"

    # Create the metalog distribution
    metalog_dist = pm.metalog(x = list_of_values,
                        probs = list_of_percentiles,
                        term_limit = term_limit,
                        bounds = [0, 100],
                        boundedness = "b")

    # Show summary of the metalog distribution
    pm.summary(m = metalog_dist)

    # Take a random sample from the metalog distribution
    array_random_sample = pm.rmetalog(
        m=metalog_dist, 
        n=10000,
        term=term_limit
    )
    
    # Convert the array to a dataframe
    df_random_sample = pd.DataFrame(array_random_sample, columns=[variable_name])
    
    # Plot the metalog distribution
    if show_distribution_plot:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=df_random_sample,
            x=variable_name,
            color=fill_color,
            alpha=fill_transparency,
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
        
        # Show the mean if requested
        if show_mean:
            # Calculate the mean
            mean = df_random_sample[variable_name].mean()
            # Show the mean as a vertical line with a label
            ax.axvline(
                x=mean,
                ymax=0.97-.02,
                color="#262626",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5
            )
            ax.text(
                x=mean, 
                y=plt.ylim()[1] * 0.97, 
                s='Mean: {:.2f}'.format(mean),
                horizontalalignment='center',
                fontname="Arial",
                fontsize=9,
                color="#262626",
                alpha=0.75
            )
        
        # Show the median if requested
        if show_median:
            # Calculate the median
            median = df_random_sample[variable_name].median()
            # Show the median as a vertical line with a label
            ax.axvline(
                x=median,
                ymax=0.90-.02,
                color="#262626",
                linestyle=":",
                linewidth=1.5,
                alpha=0.5
            )
            ax.text(
                x=median,
                y=plt.ylim()[1] * .90,
                s='Median: {:.2f}'.format(median),
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
            if caption_for_plot != None:
                # Word wrap the caption without splitting words
                if len(caption_for_plot) > 120:
                    # Split the caption into words
                    words = caption_for_plot.split(" ")
                    # Initialize the wrapped caption
                    wrapped_caption = ""
                    # Initialize the line length
                    line_length = 0
                    # Iterate through the words
                    for word in words:
                        # If the word is too long to fit on the current line, add a new line
                        if line_length + len(word) > 120:
                            wrapped_caption = wrapped_caption + "\n"
                            line_length = 0
                        # Add the word to the line
                        wrapped_caption = wrapped_caption + word + " "
                        # Update the line length
                        line_length = line_length + len(word) + 1
            else:
                wrapped_caption = ""
            
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
    
    # Return the simulated data if requested
    if return_simulated_data:
        return df_random_sample
    
    
# # Test function
# CreateMetalogDistributionFromPercentiles(
#     list_of_values=[20, 25, 30, 90],
#     list_of_percentiles=[.10, .33, .66, .90],
#     boundedness="both",
#     variable_name="Estimated Scores in Survey",
#     title_for_plot="Estimated Scores in Survey",
#     subtitle_for_plot="Based on observed scores for the 10th, 33rd, 66th, and 90th percentiles."
# )
