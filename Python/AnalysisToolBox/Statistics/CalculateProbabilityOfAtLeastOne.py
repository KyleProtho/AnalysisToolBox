# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Define function
def CalculateProbabilityOfAtLeastOne(number_of_attempts,
                                     probability_of_event,
                                     plot_probability_curve=True,
                                     # Line formatting arguments
                                     line_color="#3269a8",
                                     line_alpha=0.8,
                                     markers="o",
                                     # Text formatting arguments
                                     title_for_plot="Probability Curve",
                                     subtitle_for_plot="Shows the probability of an event occurring at least once over a given number of attempts",
                                     caption_for_plot=None,
                                     data_source_for_plot=None,
                                     x_indent=-0.11,
                                     title_y_indent=1.125,
                                     subtitle_y_indent=1.07,
                                     caption_y_indent=-0.3,
                                     # Plot formatting arguments
                                     figure_size=(8, 5)):
    # Ensure that probability of event is between 0 and 1
    if probability_of_event < 0 or probability_of_event > 1:
        raise ValueError("Probability of event must be between 0 and 1")
    
    # Calculate probability of at least one success
    probability = 1 - (1 - probability_of_event) ** number_of_attempts
    
    # Plot probability curve
    if plot_probability_curve:
        # Create array of number of attempts
        number_of_attempts_array = np.arange(1, number_of_attempts + 2)
        
        # Create array of probabilities
        probability_array = 1 - (1 - probability_of_event) ** number_of_attempts_array
        
        # Create dataframe
        dataframe = pd.DataFrame({
            "Number of Attempts": number_of_attempts_array,
            "Probability": probability_array
        })
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
    
        # Use Seaborn to create a line plot
        sns.lineplot(
            data=dataframe,
            x="Number of Attempts",
            y="Probability",
            color=line_color,
            alpha=line_alpha,
            linewidth=2,
            marker=markers,
            markersize=8,
            markeredgewidth=1,
            markeredgecolor="white",
            ax=ax
        )
        
        # Set y-axis limits to 0 and 1
        ax.set_ylim(0, 1)
        
        # Add a horizontal line at the probability of at least one success
        ax.axhline(
            y=probability,
            color=line_color,
            linestyle="--",
            alpha=line_alpha
        )
        
        # Add a label for the horizontal line
        ax.text(
            x=1,
            y=probability + 0.02,
            # Format probability to three decimal places with a percent sign
            s="Probability: " + "{:.3%}".format(probability),
            fontname="Arial",
            fontsize=10,
            color=line_color,
            horizontalalignment="left"
        )
        
        # Add a vertical line at the number of attempts
        ax.axvline(
            x=number_of_attempts,
            color=line_color,
            linestyle="--",
            alpha=line_alpha
        )
        
        # Add a label for the vertical line
        ax.text(
            x=number_of_attempts + 0.2,
            y=0.02,
            s="Attempts: " + str(number_of_attempts),
            fontname="Arial",
            fontsize=10,
            color=line_color,
            rotation=90,
            verticalalignment="bottom"
        )
        
        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        # Format tick labels to be Arial, size 9, and color #666666
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
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
        
        # Word wrap the subtitle without splitting words
        if subtitle_for_plot != None:   
            subtitle_for_plot = textwrap.fill(subtitle_for_plot, 100, break_long_words=False)
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
        
        # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
        ax.yaxis.set_label_coords(-0.1, 0.72)
        ax.yaxis.set_label_text(
            "Probability of at Least One Success",
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(
            "Number of Attempts",
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
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
    
    # Return result
    return probability


# # Test function
# # CalculateProbabilityOfAtLeastOne(number_of_attempts=10, 
# #                                  probability_of_event=0.1)
# CalculateProbabilityOfAtLeastOne(number_of_attempts=10, 
#                                  probability_of_event=0.15)
