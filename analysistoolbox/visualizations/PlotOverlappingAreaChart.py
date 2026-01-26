# Load packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotOverlappingAreaChart(dataframe,
                             value_column_name,
                             variable_column_name,
                             time_column_name,
                             # Plot formatting arguments
                             figure_size=(8, 5),
                             # Line formatting arguments
                             line_alpha=0.9,
                             color_palette="Paired",
                             # Area formatting arguments
                             fill_alpha=0.8,
                             label_variable_on_area=True,
                             label_font_size=12,
                             label_font_color="#FFFFFF",
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
                             # Plot saving arguments
                             filepath_to_save_plot=None):
    """
    Generate a formatted overlapping area chart for multi-variable time series comparison.

    This function creates a high-quality overlapping area chart using seaborn's 
    lineplot and matplotlib's fill_between. It is designed to visualize how 
    multiple categorical variables change over time, emphasizing the magnitude 
    of each variable through filled color areas. The function automatically 
    orders variables by their mean value to improve readability and provides 
    options for labeling variables directly onto the colored areas.

    Overlapping area charts are essential for:
      * Epidemiology: Comparing time-series trends of infection rates across multiple regions.
      * Healthcare: Visualizing patient census levels by hospital department over a fiscal year.
      * Intelligence Analysis: Monitoring threat activity levels across monitored geopolitical sectors.
      * Data Science: Tracking cumulative performance metrics across different model training versions.
      * Public Health: Monitoring atmospheric pollutant concentrations across several urban stations.
      * Finance: Visualizing the growth of multiple asset classes within an investment portfolio.
      * Operations: Tracking resource utilization (CPU/Memory) across different server clusters.
      * Economics: Visualizing historical unemployment rates across various demographic segments.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the time series data in long format.
    value_column_name : str
        The name of the column containing the numeric values (y-axis).
    variable_column_name : str
        The name of the categorical column used to group the data into different areas.
    time_column_name : str
        The name of the column containing the temporal data (x-axis).
    figure_size : tuple, optional
        Dimensions of the output figure as a (width, height) tuple in inches. 
        Defaults to (8, 5).
    line_alpha : float, optional
        The transparency level (alpha) for the boundary lines. Defaults to 0.9.
    color_palette : str, optional
        The name of the seaborn color palette to use for the areas. Defaults to "Paired".
    fill_alpha : float, optional
        The transparency level (alpha) for the filled areas beneath the lines. 
        Defaults to 0.8.
    label_variable_on_area : bool, optional
        Whether to place text labels for each variable directly inside their 
        respective areas. Defaults to True.
    label_font_size : int, optional
        The font size for the internal area labels. Defaults to 12.
    label_font_color : str, optional
        The hex color code for the internal area labels. Defaults to "#FFFFFF".
    number_of_x_axis_ticks : int, optional
        The maximum number of ticks to display on the horizontal time axis. 
        Defaults to None.
    x_axis_tick_rotation : int, optional
        The rotation angle (in degrees) for the x-axis tick labels. Defaults to None.
    title_for_plot : str, optional
        The primary title text displayed at the top of the chart. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the data origin, appended to the caption. Defaults to None.
    x_indent : float, optional
        Horizontal offset for the titles and captions relative to the axes. 
        Defaults to -0.127.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the grid. Defaults to 1.125.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the grid. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the grid. Defaults to -0.3.
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the overlapping area chart using matplotlib and 
        optionally saves it to disk.

    Examples
    --------
    # Epidemiology: Regional infection trends over 6 months
    import pandas as pd
    import numpy as np
    dates = pd.date_range(start="2024-01-01", periods=6, freq='M')
    epi_df = pd.DataFrame({
        'Date': np.tile(dates, 3),
        'Region': np.repeat(['East', 'West', 'Central'], 6),
        'Cases': [10, 15, 45, 60, 30, 20, 5, 8, 22, 35, 18, 12, 15, 25, 35, 45, 50, 40]
    })
    PlotOverlappingAreaChart(
        epi_df, 'Cases', 'Region', 'Date',
        title_for_plot="Viral Transmission Velocity",
        subtitle_for_plot="New confirmed cases per 100,000 residents"
    )

    # Intelligence Analysis: Threat intensity monitoring
    threat_df = pd.DataFrame({
        'Day': np.tile(np.arange(1, 11), 2),
        'Sector': np.repeat(['Sector A', 'Sector B'], 10),
        'Intensity': [2, 3, 5, 8, 7, 6, 4, 3, 2, 1, 1, 2, 2, 3, 5, 9, 8, 7, 6, 5]
    })
    PlotOverlappingAreaChart(
        threat_df, 'Intensity', 'Sector', 'Day',
        title_for_plot="Geopolitical Threat Level Analysis",
        color_palette="rocket",
        fill_alpha=0.6
    )

    # Healthcare: ICU occupancy by ward
    hosp_df = pd.DataFrame({
        'Week': np.tile(np.arange(1, 5), 2),
        'Ward': np.repeat(['Surgical ICU', 'Medical ICU'], 4),
        'Patients': [12, 14, 18, 15, 8, 10, 12, 14]
    })
    PlotOverlappingAreaChart(
        hosp_df, 'Patients', 'Ward', 'Week',
        title_for_plot="Hospital Resource Tracking",
        subtitle_for_plot="Weekly average ICU beds occupied",
        caption_for_plot="Data reflects surgical and medical intensive care units."
    )
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Order the dataframe by the mean of the value column for each variable
    averages = dataframe.groupby(variable_column_name).mean().reset_index()
    averages = averages.sort_values(by=value_column_name, ascending=False)
    dataframe[variable_column_name] = pd.Categorical(dataframe[variable_column_name], categories=averages[variable_column_name], ordered=True)
    dataframe = dataframe.sort_values(by=[variable_column_name])
    
    # Use Seaborn to create a line plot
    sns.lineplot(
        data=dataframe,
        x=time_column_name,
        y=value_column_name,
        hue=variable_column_name,
        palette=color_palette,
        alpha=line_alpha,
        ax=ax,
        legend=None if label_variable_on_area else "brief" # If label_variable_on_area is False, include a legend
    )
    
    # Create sns color palette
    sns_color_palette = sns.color_palette(color_palette)
    
    # Fill the area under the line plot for each variable
    for variable in dataframe[variable_column_name].unique():
        # Get the data for the variable
        variable_data = dataframe[dataframe[variable_column_name] == variable]
        
        # Fill the area under the line plot for the variable
        ax.fill_between(
            x=variable_data[time_column_name],
            y1=variable_data[value_column_name],
            y2=0,
            alpha=fill_alpha,
            color=sns_color_palette[dataframe[variable_column_name].unique().tolist().index(variable)] # Get the color of the variable from the sns color palette
        )
        
    # Add variable labels to the area under the line plot, if requested
    if label_variable_on_area:
        for variable in dataframe[variable_column_name].unique():
            # Get the data for the variable
            variable_data = dataframe[dataframe[variable_column_name] == variable]
            
            # Filter to records where the value is greater than 0
            variable_data = variable_data[variable_data[value_column_name] > 0]
        
            # Get the maximum value for the variable
            max_value = variable_data[value_column_name].max()
            
            # Add the variable label to the area under the line plot
            ax.text(
                x=variable_data[time_column_name].iloc[int(len(variable_data[time_column_name])/2)],
                y=max_value*0.4,
                s=variable,
                verticalalignment="bottom",
                # fontname="Arial",
                fontsize=label_font_size,
                fontweight="bold",
                color=label_font_color # Get the color of the last line in the plot
            )
    
    # Remove top and right spines, and set bottom and left spines to gray
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    
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
        subtitle_for_plot = textwrap.fill(subtitle_for_plot, 110, break_long_words=False)
        
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
    ax.yaxis.set_label_coords(-0.1, 0.84)
    ax.yaxis.set_label_text(
        value_column_name,
        # fontname="Arial",
        fontsize=10,
        color="#666666"
    )
    
    # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
    ax.xaxis.set_label_coords(0.9, -0.1)
    ax.xaxis.set_label_text(
        time_column_name,
        # fontname="Arial",
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
    
