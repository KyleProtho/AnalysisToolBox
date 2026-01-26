# Load packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotHeatmap(dataframe,
                x_axis_column_name,
                y_axis_column_name,
                value_column_name,
                # Heatmap formatting arguments
                color_palette="RdYlGn",
                color_map_transparency=0.8,
                color_map_buckets=7,
                color_map_minimum_value=None,
                color_map_maximum_value=None,
                color_map_center_value=None,
                show_legend=False,
                figure_size=(8, 6),
                border_line_width=6,
                border_line_color="#ffffff",
                square_cells=False,
                # Text formatting arguments
                show_data_labels=True,
                data_label_fontweight="normal",
                data_label_color="#262626",
                data_label_fontsize=12,
                title_for_plot=None,
                subtitle_for_plot=None,
                caption_for_plot=None,
                data_source_for_plot=None,
                title_y_indent=1.15,
                subtitle_y_indent=1.1,
                caption_y_indent=-0.17,
                # Plot saving arguments
                filepath_to_save_plot=None):
    """
    Generate a formatted heatmap to visualize three-dimensional categorical data.

    This function creates a high-quality heatmap based on a long-form DataFrame. 
    It pivot the data using specified x and y categorical columns and represents 
    the numeric values through color encoding. It supports extensive 
    customization of color schemes, data labeling, cell formatting (square vs. 
    rectangular), and professional text annotations. The function is designed 
    to handle categorical labels with automatic text wrapping.

    Heatmaps are essential for:
      * Epidemiology: Visualizing disease incidence rates across geographic regions and time periods.
      * Healthcare: Monitoring patient volume across different hospital wards and work shifts.
      * Data Science: Evaluating model performance metrics across different hyperparameter combinations.
      * Public Health: Monitoring average pollutant levels across urban districts and seasons.
      * Finance: Visualizing the performance of different asset classes across market environments.
      * Marketing: Analyzing customer engagement levels across various campaigns and segments.
      * Operations: Monitoring production efficiency across several manufacturing lines and phases.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the categorical and numeric data in long format.
    x_axis_column_name : str
        The name of the column to be used for the horizontal axis categories.
    y_axis_column_name : str
        The name of the column to be used for the vertical axis categories.
    value_column_name : str
        The name of the column containing the numeric values to be color-encoded.
    color_palette : str or list, optional
        The seaborn/matplotlib color map or list of colors. Defaults to "RdYlGn".
    color_map_transparency : float, optional
        The transparency level (alpha) of the heatmap cells. Defaults to 0.8.
    color_map_buckets : int, optional
        The number of discrete color levels for the heatmap scale. Defaults to 7.
    color_map_minimum_value : float, optional
        The absolute minimum value for the color scale. If None, it is 
        inferred from the data. Defaults to None.
    color_map_maximum_value : float, optional
        The absolute maximum value for the color scale. If None, it is 
        inferred from the data. Defaults to None.
    color_map_center_value : float, optional
        The value representing the center of the color scale (useful for 
        diverging colormaps). If None, the median is used. Defaults to None.
    show_legend : bool, optional
        Whether to display the color bar legend mapping values to colors. 
        Defaults to False.
    figure_size : tuple, optional
        The dimensions of the output figure as a (width, height) tuple in inches. 
        Defaults to (8, 6).
    border_line_width : int, optional
        The width of the lines separating individual heatmap cells. Defaults to 6.
    border_line_color : str, optional
        The hex color code for the cell border lines. Defaults to "#ffffff".
    square_cells : bool, optional
        Whether to force each heatmap cell to be perfectly square. Defaults to False.
    show_data_labels : bool, optional
        Whether to display numeric text labels inside each heatmap cell. Defaults to True.
    data_label_fontweight : str, optional
        The font weight for the internal cell labels (e.g., "bold", "normal"). 
        Defaults to "normal".
    data_label_color : str, optional
        The hex color code for the internal cell labels. Defaults to "#262626".
    data_label_fontsize : int, optional
        The font size for the internal cell labels. Defaults to 12.
    title_for_plot : str, optional
        The primary title text at the top of the figure. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the data source, appended to the caption. Defaults to None.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the grid. Defaults to 1.15.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the grid. Defaults to 1.1.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the grid. Defaults to -0.17.
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the heatmap using matplotlib and optionally saves 
        it to disk.

    Examples
    --------
    # Epidemiology: Infection rates by region and month
    import pandas as pd
    epi_df = pd.DataFrame({
        'Region': ['North', 'North', 'South', 'South'],
        'Month': ['Jan', 'Feb', 'Jan', 'Feb'],
        'Rate': [12.5, 14.2, 8.5, 9.1]
    })
    PlotHeatmap(
        epi_df, 'Month', 'Region', 'Rate',
        title_for_plot="Regional Infection Velocity",
        color_palette="YlOrRd"
    )

    # Healthcare: Ward occupancy levels by shift
    hosp_df = pd.DataFrame({
        'Ward': ['ICU', 'ICU', 'Emergency', 'Emergency'],
        'Shift': ['Day', 'Night', 'Day', 'Night'],
        'Occupancy': [95, 80, 75, 90]
    })
    PlotHeatmap(
        hosp_df, 'Shift', 'Ward', 'Occupancy',
        title_for_plot="Hospital Facility Utilization",
        data_label_fontweight="bold",
        color_palette="Blues"
    )
    """
    
    # Convert dataframe to wide-form dataframe, using x_axis_column_name as the index and y_axis_column_name as the columns
    dataframe = dataframe.pivot(
        index=x_axis_column_name, 
        columns=y_axis_column_name,
        values=value_column_name
    )
    
    # Transpose dataframe
    dataframe = dataframe.transpose()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Adjust color map minimum and maximum values if they are not provided
    if color_map_minimum_value == None:
        color_map_minimum_value = dataframe.min().min()
    if color_map_maximum_value == None:
        color_map_maximum_value = dataframe.max().max()
    if color_map_center_value == None:
        color_map_center_value = dataframe.median().median()
        
    # Convert color palette to a list if it is not already a list
    if type(color_palette) != list:
        color_palette = sns.color_palette(color_palette, color_map_buckets)
        # color_palette.set_bad(color='white', alpha=color_palette_transparency)
    
    # Plot contingency table in a heatmap using seaborn
    ax = sns.heatmap(
        data=dataframe,
        cmap=color_palette,
        center=color_map_center_value,
        vmin=color_map_minimum_value,
        vmax=color_map_maximum_value,
        annot=show_data_labels,
        linewidths=border_line_width,
        linecolor=border_line_color,
        cbar=show_legend,
        square=square_cells,
        ax=ax,
        annot_kws={
            "fontweight": data_label_fontweight,
            # "fontname": "Arial", 
            "fontsize": data_label_fontsize,
            "color": data_label_color,
        }
    )
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(y_axis_column_name, 30))  # String wrap the variable name
    ax.set_ylabel(
        wrapped_variable_name, 
        fontsize=12, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in y_tick_labels]
    ax.set_yticklabels(
        wrapped_y_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Wrap x axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(x_axis_column_name, 30))  # String wrap the variable name
    ax.set_xlabel(
        wrapped_variable_name, 
        fontsize=12, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Format and wrap x axis tick labels using textwrap
    x_tick_labels = ax.get_xticklabels()
    wrapped_x_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in x_tick_labels]
    ax.set_xticklabels(
        wrapped_x_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Set the x indent of the plot titles and captions
    # Get longest y tick label
    longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
    if len(longest_y_tick_label) >= 30:
        x_indent = -0.3
    else:
        x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)
        
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

