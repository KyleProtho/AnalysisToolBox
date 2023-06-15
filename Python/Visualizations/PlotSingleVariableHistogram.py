import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotSingleVariableHistogram(dataframe,
                                quantitative_variable,
                                fill_color="#3269a8",
                                fill_transparency=0.6,
                                title_for_plot=None,
                                subtitle_for_plot=None,
                                caption_for_plot=None,
                                data_source_for_plot=None,
                                show_mean=True,
                                show_median=True,
                                figure_size=(8, 6),
                                show_y_axis=False,
                                title_y_indent=1.1,
                                subtitle_y_indent=1.05,
                                caption_y_indent=-0.15):
    """This function creates a histogram of a single quantitative variable.

    Args:
        dataframe (_type_): The dataframe containing the data to be plotted.
        quantitative_variable (_type_): The name of the column in the dataframe to be plotted.
        fill_color (str, optional): The color to fill the histogram bars. Defaults to "#3269a8".
        fill_transparency (float, optional): The transparency of the histogram bars. Defaults to 0.6.
        title_for_plot (_type_, optional): The title of the plot. Defaults to None.
        subtitle_for_plot (_type_, optional): The subtitle of the plot. Defaults to None.
        caption_for_plot (_type_, optional): The caption of the plot. Defaults to None.
        data_source_for_plot (_type_, optional): The data source of the plot. Defaults to None.
        show_mean (bool, optional): Whether to show the mean on the plot. Defaults to True.
        show_median (bool, optional): Whether to show the median on the plot. Defaults to True.
        figure_size (tuple, optional): The size of the plot. Defaults to (8, 6).
        show_y_axis (bool, optional): Whether to show the y-axis. Defaults to False.
        title_y_indent (float, optional): The vertical indent of the title. Defaults to 1.1.
        subtitle_y_indent (float, optional): The vertical indent of the subtitle. Defaults to 1.05.
        caption_y_indent (float, optional): The vertical indent of the caption. Defaults to -0.15.
    """
    
    # Check that the column exists in the dataframe.
    if quantitative_variable not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(quantitative_variable))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create histogram using seaborn
    sns.histplot(
        data=dataframe,
        x=quantitative_variable,
        color=fill_color,
        alpha=fill_transparency,
    )
    
    # Remove all spines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
        mean = dataframe[quantitative_variable].mean()
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
        median = dataframe[quantitative_variable].median()
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


# Test function
from sklearn import datasets
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# PlotSingleVariableHistogram(
#     dataframe=iris,
#     quantitative_variable="sepal length (cm)",
#     title_for_plot="Sepal Length (cm)",
#     subtitle_for_plot="Iris Dataset"
# )
PlotSingleVariableHistogram(
    dataframe=iris,
    quantitative_variable="sepal length (cm)",
    title_for_plot="Sepal Length (cm)",
    subtitle_for_plot="Iris Dataset",
    caption_for_plot="This is a caption that is long enough to wrap onto multiple lines. This is a caption that is long enough to wrap onto multiple lines. This is a caption that is long enough to wrap onto multiple lines.",
    data_source_for_plot="https://archive.ics.uci.edu/ml/datasets/iris"
)
