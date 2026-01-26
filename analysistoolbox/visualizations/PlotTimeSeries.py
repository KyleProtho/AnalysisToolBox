# Load packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotTimeSeries(dataframe,
                   value_column_name,
                   time_column_name,
                   grouping_column_name=None,
                   # Line formatting arguments
                   line_color="#3269a8",
                   line_alpha=0.8,
                   color_palette="Set2",
                   show_legend=True,
                   # Marker formatting arguments
                   marker_size_column_name=None,
                   marker_size_normalization=None,
                   # Text formatting arguments
                   number_of_x_axis_ticks=None,
                   x_axis_tick_rotation=None,
                   title_for_plot=None,
                   subtitle_for_plot=None,
                   caption_for_plot=None,
                   data_source_for_plot=None,
                   x_indent=-0.127,
                   title_y_indent=1.125,
                   subtitle_y_indent=1.05,
                   caption_y_indent=-0.3,
                   # Plot formatting arguments
                   figure_size=(8, 5),
                   # Plot saving arguments
                   filepath_to_save_plot=None):
    """
    Generate a formatted time series line plot with optional grouping and markers.

    This function creates a high-quality line plot using seaborn, specifically 
    tailored for visualizing temporal data distributions. It supports grouping 
    by categorical variables (using color), overlaying scatter markers with 
    variable sizing, and professional formatting for titles, subtitles, and 
    captions. The function automatically handles x-axis tick density and 
    rotation to ensure long time-series data remains readable.

    Time series plots are essential for:
      * Epidemiology: Tracking the daily count of new infection cases during an epidemic.
      * Healthcare: Monitoring patient vitals (e.g., heart rate) over the course of a hospital stay.
      * Intelligence Analysis: Visualizing changes in identified troop movements over a tactical cycle.
      * Data Science: Monitoring model performance metrics across sequential training epochs.
      * Public Health: Tracking seasonal fluctuations in urban air quality indices.
      * Finance: Analyzing historical price movements of various assets to identify trends.
      * Operations: Monitoring server response times or throughput during peak traffic periods.
      * Marketing: Tracking weekly conversion rates following a multi-channel campaign launch.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the temporal data, numeric values, 
        and optional grouping variables.
    value_column_name : str
        The name of the column containing the numeric values to be plotted on 
        the vertical y-axis.
    time_column_name : str
        The name of the column containing the temporal data (e.g., dates or 
        sessions) for the x-axis.
    grouping_column_name : str, optional
        The name of a categorical column used to group the data into multiple 
        colored lines. Defaults to None.
    line_color : str, optional
        The hex color code for the time-series line when no grouping is applied. 
        Defaults to "#3269a8".
    line_alpha : float, optional
        The transparency level (alpha) of the line and associated markers 
        (0 to 1). Defaults to 0.8.
    color_palette : str, optional
        The name of the seaborn color palette to use when grouping by a 
        categorical variable. Defaults to "Set2".
    show_legend : bool, optional
        Whether to display the legend for grouping categories. Defaults to True.
    marker_size_column_name : str, optional
        The name of an additional numeric column to control the size of 
        optional scatter markers. Defaults to None.
    marker_size_normalization : tuple, optional
        A (min, max) tuple representing the scaling range for the markers. 
        Defaults to None.
    number_of_x_axis_ticks : int, optional
        The maximum number of ticks to display on the horizontal time axis 
        to avoid overcrowding. Defaults to None.
    x_axis_tick_rotation : int, optional
        The rotation angle (in degrees) for the x-axis tick labels. Defaults to None.
    title_for_plot : str, optional
        The primary title text displayed at the top of the figure. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    x_indent : float, optional
        Horizontal offset for the titles and captions relative to the axes. 
        Defaults to -0.127.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the grid. Defaults to 1.125.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the grid. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the grid. Defaults to -0.3.
    figure_size : tuple, optional
        Dimensions of the output figure as a (width, height) tuple in inches. 
        Defaults to (8, 5).
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the time series plot using matplotlib and 
        optionally saves it to disk.

    Examples
    --------
    # Epidemiology: Daily infection monitoring
    import pandas as pd
    import numpy as np
    dates = pd.date_range(start="2024-01-01", periods=30)
    epi_df = pd.DataFrame({
        'Date': dates,
        'Daily Cases': np.random.randint(10, 100, 30)
    })
    PlotTimeSeries(
        epi_df, 'Daily Cases', 'Date',
        title_for_plot="Pandemic Surveillance Dashboard",
        subtitle_for_plot="Daily confirmed infections monitor"
    )

    # Intelligence Analysis: Monitoring troop movements
    intel_df = pd.DataFrame({
        'Date': pd.date_range(start="2024-03-01", periods=10),
        'Movements': [12, 15, 8, 25, 45, 30, 20, 55, 60, 42],
        'Sector': ['Alpha'] * 5 + ['Bravo'] * 5
    })
    PlotTimeSeries(
        intel_df, 'Movements', 'Date', grouping_column_name='Sector',
        title_for_plot="Sector Troop Transition Activity",
        subtitle_for_plot="Observed major asset movements across monitored sectors"
    )

    # Healthcare: Patient vital sign tracking
    vitals_df = pd.DataFrame({
        'Hour': np.arange(1, 25),
        'HR': np.random.normal(75, 5, 24),
        'Fever Intensity': np.random.uniform(0, 1, 24)
    })
    PlotTimeSeries(
        vitals_df, 'HR', 'Hour',
        marker_size_column_name='Fever Intensity',
        title_for_plot="Patient Vitals Monitor",
        caption_for_plot="Markers are sized by normalized fever intensity measurements."
    )
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Use Seaborn to create a line plot
    if grouping_column_name is None:
        sns.lineplot(
            data=dataframe,
            x=time_column_name,
            y=value_column_name,
            color=line_color,
            alpha=line_alpha,
            ax=ax
        )
    else:
        sns.lineplot(
            data=dataframe,
            x=time_column_name,
            y=value_column_name,
            hue=grouping_column_name,
            palette=color_palette,
            alpha=line_alpha,
            ax=ax
        )
        
    # Add scatterplot markers if a column name is provided
    if marker_size_column_name is not None:
        if grouping_column_name is None:
            sns.scatterplot(
                data=dataframe,
                x=time_column_name,
                y=value_column_name,
                color=line_color,
                size=marker_size_column_name,
                size_norm=marker_size_normalization,
                alpha=line_alpha,
                ax=ax,
                linewidth=0.5,
                edgecolor="white",
                legend=False
            )
        else:
            sns.scatterplot(
                data=dataframe,
                x=time_column_name,
                y=value_column_name,
                hue=grouping_column_name,
                size=marker_size_column_name,
                size_norm=marker_size_normalization,
                palette=color_palette,
                alpha=line_alpha,
                ax=ax,
                linewidth=0.5,
                edgecolor="white",
                legend=False
            )
    
    # Remove top and right spines, and set bottom and left spines to gray
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    
    # If show_legend is False, remove the legend
    if show_legend == False:
        ax.legend_.remove()
    
    # Set the number of ticks on the x-axis
    if number_of_x_axis_ticks is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(number_of_x_axis_ticks))
        
    # Rotate x-axis tick labels
    if x_axis_tick_rotation is not None:
        plt.xticks(rotation=x_axis_tick_rotation)

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
        # fontname="Arial",
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
            # fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
    
    # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
    ax.yaxis.set_label_coords(-0.1, 0.99)
    ax.yaxis.set_label_text(
        textwrap.fill(value_column_name, 30, break_long_words=False),
        # fontname="Arial",
        fontsize=10,
        color="#666666",
        ha='right',
    )
    
    # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
    ax.xaxis.set_label_coords(0.99, -0.1)
    ax.xaxis.set_label_text(
        textwrap.fill(time_column_name, 30, break_long_words=False),
        # fontname="Arial",
        fontsize=10,
        color="#666666",
        ha='right',
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
    
