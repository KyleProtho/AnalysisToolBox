# Load packages
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymetalog as pm
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def CreateMetalogDistribution(dataframe,
                              variable,
                              # Metalog distribution arguments
                              lower_bound=None,
                              upper_bound=None,
                              learning_rate=.01,
                              term_maximum=9,
                              term_minimum=2,
                              term_for_random_sample=None,
                              number_of_samples=10000,
                              show_comparison_plot=True,
                              show_summary=True,
                              sample_format='dataframe',
                              # Histogram formatting arguments
                              fill_color="#999999",
                              fill_transparency=0.6,
                              show_mean=True,
                              show_median=True,
                              # Text formatting arguments
                              title_for_plot="Metalog Distribution",
                              subtitle_for_plot="Showing the metalog distribution of the variable",
                              caption_for_plot=None,
                              data_source_for_plot=None,
                              show_y_axis=False,
                              title_y_indent=1.1,
                              subtitle_y_indent=1.05,
                              caption_y_indent=-0.15,
                              # Plot formatting arguments
                              figure_size=(8, 6)):
    """_summary_
    This function creates a metalog distribution from a dataset and variable.
    
    Args:
        dataframe (_type_): Pandas dataframe containing the dataset.
        variable (str): Name of the variable to create the metalog distribution from.
        lower_bound (int or float, optional): The lower bound of the metalog distribution. Defaults to None.
        upper_bound (int or float, optional): The upper bound of the metalog distribution. Defaults to None.
        learning_rate (float, optional): The learning rate to use when creating the metalog distribution. Defaults to .01.
        term_maximum (int, optional): The maximum number of terms to use when creating the metalog distribution. Defaults to 9.
        term_minimum (int, optional): The minimum number of terms to use when creating the metalog distribution. Defaults to 2.
        show_comparison_plot (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    # Ensure that sample_format is either 'dataframe' or 'array'
    if sample_format not in ['dataframe', 'array']:
        raise ValueError("sample_format must be either 'dataframe' or 'array'.")
    
    # Select necessary columns from the dataframe
    dataframe = dataframe[[variable]]
    
    # Filter NA, None, and infinite values from the dataframe
    dataframe = dataframe[dataframe[variable].notna()]
    dataframe = dataframe[dataframe[variable] != None]
    dataframe = dataframe[dataframe[variable] != np.inf]
    
    # Extract values from the dataframe
    arr_variable = dataframe[variable].values
    
    # Create a metalog distribution
    if lower_bound is None and upper_bound is None:
        metalog_dist = pm.metalog(
            x=arr_variable,
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    elif lower_bound is not None and upper_bound is None:
        metalog_dist = pm.metalog(
            x=arr_variable,
            boundedness='sl',
            bounds=[lower_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    elif lower_bound is None and upper_bound is not None:
        metalog_dist = pm.metalog(
            x=arr_variable,
            boundedness='su',
            bounds=[upper_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    else:
        metalog_dist = pm.metalog(
            x=arr_variable,
            boundedness='b',
            bounds=[lower_bound, upper_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
        
    # Show summary of the metalog distribution, if requested
    if show_summary:
        pm.summary(m = metalog_dist)
        
    # Get the maximum number of terms used in the metalog distribution that is valid
    if term_for_random_sample is None:
        term_for_random_sample = metalog_dist.term_limit
        
    # Randomly select 10000 values from the metalog distribution
    arr_metalog = pm.rmetalog(
        metalog_dist, 
        n=number_of_samples, 
        term=term_for_random_sample, 
        generator="hdr"
    )
    
    # Convert the metalog distribution to a dataframe
    metalog_df = pd.DataFrame(arr_metalog, columns=[variable])
    
    # # To-Do: Add a comparison plot of the metalog distribution and the data
    if show_comparison_plot:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=metalog_df,
            x=variable,
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
            mean = metalog_df[variable].mean()
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
            median = metalog_df[variable].median()
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
        
    # Return the metalog distribution
    if sample_format == 'dataframe':
        return metalog_df
    else:
        return arr_metalog


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# iris = iris[iris['species'] == 0]
# # petal_legnth_dist = CreateMetalogDistribution(
# #     dataframe=iris,
# #     variable='petal length (cm)',
# # )
# # petal_legnth_dist = CreateMetalogDistribution(
# #     dataframe=iris,
# #     variable='petal length (cm)',
# #     upper_bound=7,
# # )
# petal_legnth_dist = CreateMetalogDistribution(
#     dataframe=iris,
#     variable='petal length (cm)',
#     lower_bound=0,
#     upper_bound=3
# )
