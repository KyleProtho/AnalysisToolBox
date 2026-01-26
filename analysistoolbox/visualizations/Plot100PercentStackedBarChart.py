# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def Plot100PercentStackedBarChart(dataframe,
                                  group_column_name,
                                  value_column_name,
                                  target_value_column_name=None,
                                  # Plot formatting arguments
                                  background_color='#e8e8ed',
                                  color_palette="Set1",
                                  fill_color=None,
                                  fill_transparency=0.8,
                                  figure_size=(8, 6),
                                  # Text formatting arguments
                                  data_label_fontsize=10,
                                  title_for_plot=None,
                                  subtitle_for_plot=None,
                                  caption_for_plot=None,
                                  data_source_for_plot=None,
                                  title_y_indent=1.15,
                                  subtitle_y_indent=1.1,
                                  caption_y_indent=-0.15,
                                  # Plot saving arguments
                                  filepath_to_save_plot=None):
    """
    Generate a 100% horizontal stacked bar chart visualizing completion or composition.

    This function creates a horizontal bar chart where each bar represents a group, 
    and the filled portion represents either a percentage of a total sum across 
    all groups or the progress toward a specific target value for that group. 
    The remaining portion of the bar is rendered in a background color to 
    illustrate the deficit or remaining capacity. The function handles text wrapping 
    for long group names and provides extensive formatting options for titles, 
    subtitles, and captions.

    100% stacked bar charts are essential for:
      * Epidemiology: Comparing vaccination coverage rates across multiple geographic regions.
      * Intelligence Analysis: Visualizing the completion status of intelligence collection requirements.
      * Healthcare: Monitoring medication adherence levels or patient satisfaction benchmarks.
      * Project Management: Tracking progress toward milestones across different project tracks.
      * Public Health: Monitoring hospital bed occupancy or ICU capacity by facility.
      * Sales: Visualizing revenue achieved vs. target quotas for regional sales teams.
      * Data Science: Monitoring data labeling progress or model training iterations.
      * Quality Control: Comparing the proportion of units passing inspection across production lines.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the data to be visualized. Must contain 
        one row per group.
    group_column_name : str
        The name of the column containing the category or group labels for the bars.
    value_column_name : str
        The name of the column containing the current or numerator values.
    target_value_column_name : str, optional
        The name of the column containing the target or denominator values. If 
        None, the percentage is calculated as the value's share of the total 
        sum across all groups. Defaults to None.
    background_color : str, optional
        The hex color code used for the empty/remaining portion of the bars. 
        Defaults to '#e8e8ed'.
    color_palette : str, optional
        The seaborn color palette used for the filled portion of the bars if 
        `fill_color` is not specified. Defaults to "Set1".
    fill_color : str, optional
        A specific hex color code to use for all bars. If provided, overrides 
        `color_palette`. Defaults to None.
    fill_transparency : float, optional
        The transparency level (alpha) of the bar fill, ranging from 0 to 1. 
        Defaults to 0.8.
    figure_size : tuple, optional
        A tuple of (width, height) in inches for the plot. Defaults to (8, 6).
    data_label_fontsize : int, optional
        The font size for axis and group labels. Defaults to 10.
    title_for_plot : str, optional
        The primary title text displayed at the top of the figure. Defaults to None.
    subtitle_for_plot : str, optional
        The descriptive subtitle text displayed below the title. Defaults to None.
    caption_for_plot : str, optional
        Descriptive text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Data attribution text appended to the caption. Defaults to None.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the axes. Defaults to 1.15.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the axes. Defaults to 1.1.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the axes. Defaults to -0.15.
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the plot using matplotlib and optionally saves 
        it to disk.

    Examples
    --------
    # Epidemiology: Vaccination coverage across different provinces
    import pandas as pd
    prov_data = pd.DataFrame({
        'Province': ['North', 'South', 'East', 'West'],
        'Vaccinated': [8500, 7200, 9100, 6800],
        'Population': [10000, 10000, 10000, 10000]
    })
    Plot100PercentStackedBarChart(
        prov_data, 'Province', 'Vaccinated', 'Population',
        title_for_plot="COVID-19 Vaccination Coverage",
        subtitle_for_plot="Percentage of population fully vaccinated by province",
        color_palette="viridis"
    )

    # Intelligence Analysis: Progress of collection requirements by source type
    intel_df = pd.DataFrame({
        'Source': ['SIGINT', 'HUMINT', 'OSINT', 'GEOINT'],
        'Items Collected': [150, 45, 300, 80],
        'Target Items': [200, 100, 500, 100]
    })
    Plot100PercentStackedBarChart(
        intel_df, 'Source', 'Items Collected', 'Target Items',
        title_for_plot="Intelligence Collection Progress",
        fill_color="#2c3e50",
        caption_for_plot="Mission cycle 2024-Q1 data summary."
    )

    # Healthcare: Patient satisfaction scores by ward
    satisfaction_df = pd.DataFrame({
        'Ward': ['Emergency', 'ICU', 'Maternity', 'Pediatrics'],
        'Satisfied': [45, 12, 38, 55]
    })
    Plot100PercentStackedBarChart(
        satisfaction_df, 'Ward', 'Satisfied',
        title_for_plot="Patient Satisfaction Composition",
        subtitle_for_plot="Proportion of total satisfied patients by ward",
        color_palette="RdBu"
    )
    """
    
    # Ensure that the number of unique values in the 'Group' column is equal to the number of rows in the DataFrame
    if len(dataframe[group_column_name].unique()) != len(dataframe):
        raise ValueError("The number of unique values in the 'Group' column must be equal to the number of rows in the DataFrame.")
    
    # Calculate the percentage of the Current Value out of the Target Value
    if target_value_column_name != None:
        dataframe['Percentage'] = dataframe[value_column_name] / dataframe[target_value_column_name] * 100
        dataframe['Remaining'] = 100 - dataframe['Percentage']
    else:
        dataframe['Percentage'] = dataframe[value_column_name] / dataframe[value_column_name].sum() * 100
        dataframe['Remaining'] = 100 - dataframe['Percentage']
    
    # Sort the DataFrame based on the 'Group' column to match the user's example
    dataframe.sort_values(by=group_column_name, inplace=True)
    
    # Short by the 'Percentage' column to make the plot look better
    dataframe.sort_values(by='Percentage', inplace=True)
    
    # Set the figure size for better clarity
    plt.figure(figsize=figure_size)
    
    # If fill_color is not provided, use the color_palette
    if fill_color == None:
        # Create a color palette
        color_palette = sns.color_palette(color_palette, len(dataframe))
        
        # Create a list of colors
        colors = [color_palette[i] for i in range(len(dataframe))]
    else:
        # Create a list of colors
        colors = [fill_color for i in range(len(dataframe))]
        
    # Create subplot
    ax = plt.subplot(111)
    
    # Plot the 'Percentage' bar horizontally
    ax.barh(
        dataframe[group_column_name], 
        dataframe['Percentage'], 
        color=colors,
        edgecolor='white',
        alpha=fill_transparency
    )
    
    # Add the 'Remaining' bar horizontally behind the 'Percentage' bar
    ax.barh(
        dataframe[group_column_name], 
        dataframe['Remaining'], 
        left=dataframe['Percentage'], 
        color=background_color, 
        edgecolor='white',
    )
    
    # Set the x-axis to show percentages
    plt.xticks(np.arange(0, 101, 10))
    
    # Limit the x-axis to 100%
    plt.xlim(0, 100)
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(group_column_name, 40, break_long_words=False))  # String wrap the variable name
    plt.ylabel(wrapped_variable_name, fontsize=10, color="#262626")
    
    # Format and wrap y axis tick labels using textwrap
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label, 40, break_long_words=False)) for label in dataframe[group_column_name].tolist()]
    plt.yticks(
        dataframe['Group'], 
        wrapped_y_tick_labels, 
        fontsize=10, 
        color="#262626"
    )
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
    # Change x-axis colors to "#666666"
    plt.tick_params(axis='x', colors="#666666")
    
    # Remove the top, right, and left spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
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

