# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def PlotRiskTolerance(sip,
                      risk_tolerance,
                      # Risk tolerance arguments
                      lower_is_worse=True,
                      variable_name="Outcome",
                      observed_value=None,
                      risk_tolerance_label="Tolerance",
                      # Histogram formatting arguments
                      fill_color="#999999",
                      fill_transparency=0.6,
                      show_mean=True,
                      show_median=True,
                      # Text formatting arguments
                      title_for_plot="Risk Tolerance",
                      subtitle_for_plot="Show the simulated outcomes that are worse than the risk tolerance.",
                      caption_for_plot=None,
                      data_source_for_plot=None,
                      show_y_axis=False,
                      title_y_indent=1.1,
                      subtitle_y_indent=1.05,
                      caption_y_indent=-0.15,
                      # Plot formatting arguments
                      figure_size=(8, 6)):
    # Ensure that risk tolerance is numeric
    if not isinstance(risk_tolerance, (int, float)):
        raise ValueError("Risk tolerance must be numeric.")
    
    # Ensure that observed_value is numeric if it is provided
    if observed_value != None and not isinstance(observed_value, (int, float)):
        raise ValueError("Observed value must be numeric.")
    
    # Create dataframe with only the outcome variable and the SIP
    dataframe = pd.DataFrame(sip, columns=[variable_name])
    
    # Create risk flag column name
    if lower_is_worse:
        risk_flag_col_name = "Is Below Risk Tolerance"
    else:
        risk_flag_col_name = "Is Above Risk Tolerance"
    
    # Add flag for whether the outcome is worse than the risk tolerance
    if lower_is_worse:
        dataframe[risk_flag_col_name] = dataframe[variable_name] < risk_tolerance
    else:
        dataframe[risk_flag_col_name] = dataframe[variable_name] > risk_tolerance
        
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create histogram using seaborn
    sns.histplot(
        data=dataframe,
        x=variable_name,
        color=fill_color,
        alpha=fill_transparency,
        hue=risk_flag_col_name,
        palette={True: "#FF0000", False: "#999999"}
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
    
    # Calculate the percentage of outcomes that are worse than the risk tolerance
    risk_probability = dataframe[risk_flag_col_name].mean()
    
    # Show the risk tolerance as a vertical line with a label and probability
    ax.axvline(
        x=risk_tolerance,
        ymax=0.97-.02,
        color="#262626",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5
    )
    ax.text(
        x=risk_tolerance, 
        y=plt.ylim()[1] * 0.97, 
        s=risk_tolerance_label+': {:.2f}'.format(risk_tolerance)+' ({:.1%})'.format(risk_probability),
        horizontalalignment='center',
        fontname="Arial",
        fontsize=9,
        color="#262626",
        alpha=0.75
    )
    
    # Show the oberseved value as a vertical line with a label
    if observed_value != None:
        ax.axvline(
            x=observed_value,
            ymax=0.90-.02,
            color="#262626",
            linewidth=2,
            alpha=0.8
        )
        ax.text(
            x=observed_value, 
            y=plt.ylim()[1] * .90,
            s=variable_name+': {:.2f}'.format(observed_value),
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
    

# # Test function
# import numpy as np
# # PlotRiskTolerance(sip=np.random.normal(0, 1, 10000),
# #                   risk_tolerance=-0.5)
# PlotRiskTolerance(sip=np.random.normal(0, 1, 10000),
#                   risk_tolerance=-0.5,
#                   observed_value=-0.75)
# # PlotRiskTolerance(sip=np.random.normal(0, 1, 10000),
# #                   risk_tolerance=-0.5,
# #                   lower_is_worse=False)
