# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotSingleVariableHistogram(dataframe,
                                value_column_name,
                                # Histogram formatting arguments
                                fill_color="#999999",
                                fill_transparency=0.6,
                                show_mean=True,
                                show_median=True,
                                # Text formatting arguments
                                title_for_plot=None,
                                subtitle_for_plot=None,
                                caption_for_plot=None,
                                data_source_for_plot=None,
                                show_y_axis=False,
                                title_y_indent=1.1,
                                subtitle_y_indent=1.05,
                                caption_y_indent=-0.15,
                                # Plot formatting arguments
                                figure_size=(8, 6),
                                # Plot saving arguments
                                filepath_to_save_plot=None):
    """
    Generate a formatted histogram to visualize the distribution of a single numeric variable.

    This function creates a high-quality histogram using seaborn to represent the 
    underlying frequency distribution of a continuous numeric variable. It 
    supports optional central tendency indicators (mean and median) through 
    vertical reference lines and annotations. The plot features professional 
    formatting, including customizable titles, subtitles, and captions with 
    automatic word wrapping.

    Single-variable histograms are essential for:
      * Epidemiology: Visualizing the distribution of patient ages in a disease outbreak.
      * Healthcare: Analyzing the distribution of systolic blood pressure readings across a population.
      * Intelligence Analysis: Monitoring the distribution of travel distances for suspected illicit transport.
      * Data Science: Inspecting numeric features for skewness, outliers, and normalization requirements.
      * Public Health: tracking the distribution of daily water consumption per capita.
      * Finance: Examining the distribution of trade transaction sizes in a specific portfolio.
      * Project Management: Analyzing the distribution of task completion times across a project lifecycle.
      * Social Science: Visualizing the distribution of household income levels in a surveyed region.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the numeric variable to analyze.
    value_column_name : str
        The name of the continuous numeric column to be plotted on the x-axis.
    fill_color : str, optional
        The hex color code for the histogram bars. Defaults to "#999999".
    fill_transparency : float, optional
        The transparency level (alpha) of the histogram bars (0 to 1). Defaults to 0.6.
    show_mean : bool, optional
        Whether to overlay a dashed vertical line and text label indicating the 
        arithmetic mean of the data. Defaults to True.
    show_median : bool, optional
        Whether to overlay a dotted vertical line and text label indicating the 
        median of the data. Defaults to True.
    title_for_plot : str, optional
        The primary title text displayed at the top of the chart. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    show_y_axis : bool, optional
        Whether to display the vertical frequency (count) axis. Defaults to False.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the axes. Defaults to 1.1.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the axes. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the axes. Defaults to -0.15.
    figure_size : tuple, optional
        Dimensions of the output figure as a (width, height) tuple in inches. Defaults to (8, 6).
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the histogram using matplotlib and optionally 
        saves it to disk.

    Examples
    --------
    # Epidemiology: Age distribution of study participants
    import pandas as pd
    import numpy as np
    epi_df = pd.DataFrame({'Age': np.random.normal(45, 12, 500)})
    PlotSingleVariableHistogram(
        epi_df, 'Age', 
        fill_color="#27ae60",
        title_for_plot="Participant Age Distribution",
        subtitle_for_plot="Samples collected from regional demographic survey (N=500)"
    )

    # Intelligence Analysis: Suspected transport travel distances
    intel_df = pd.DataFrame({'Distance': np.random.lognormal(2.5, 0.5, 1000)})
    PlotSingleVariableHistogram(
        intel_df, 'Distance',
        fill_color="#2c3e50",
        show_median=True,
        title_for_plot="Logistics Anomaly Detection",
        subtitle_for_plot="Analysis of travel distances for identified transport vehicles"
    )

    # Healthcare: Patient vitals monitoring
    hosp_df = pd.DataFrame({'BP': np.random.normal(120, 15, 200)})
    PlotSingleVariableHistogram(
        hosp_df, 'BP',
        show_y_axis=True,
        title_for_plot="Population Blood Pressure Profile",
        caption_for_plot="Dashed line represents the mean; dotted line represents the median."
    )
    """
    
    # Check that the column exists in the dataframe.
    if value_column_name not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(value_column_name))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create histogram using seaborn
    sns.histplot(
        data=dataframe,
        x=value_column_name,
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
        mean = dataframe[value_column_name].mean()
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
            # fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
    
    # Show the median if requested
    if show_median:
        # Calculate the median
        median = dataframe[value_column_name].median()
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
            # fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
    
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        # fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        # fontname="Arial",
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
    # plt.xticks(fontname='Arial')
    
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
            # fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.transAxes
        )
    
    # If filepath_to_save_plot is provided, save the plot
    if filepath_to_save_plot != None:
        # Ensure that the filepath ends with '.png' or '.jpg'
        if not filepath_to_save_plot.endswith('.png') and not filepath_to_save_plot.endswith('.jpg'):
            raise ValueError("The filepath to save the plot must end with '.png' or '.jpg'.")
        
        # Save plot
        plt.savefig(
            filepath_to_save_plot,
            bbox_inches="tight"
        )
       
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

