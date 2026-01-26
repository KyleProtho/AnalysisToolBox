# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotClusteredBarChart(dataframe,
                          grouping_column_name_1, 
                          grouping_column_name_2,
                          value_column_name,
                          # Plot formatting arguments
                          color_palette="Set1",
                          fill_transparency=0.8,
                          display_order_list=None,
                          figure_size=(6, 6),
                          show_legend=True,
                          decimal_places_for_data_label=0,
                          # Text formatting arguments
                          title_for_plot=None,
                          subtitle_for_plot=None,
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          x_indent=-0.04,
                          title_y_indent=1.15,
                          subtitle_y_indent=1.1,
                          caption_y_indent=-0.15,
                          # Plot saving arguments
                          filepath_to_save_plot=None):
    """
    Generate a formatted clustered bar chart for multi-category data comparison.

    This function creates a high-quality clustered (grouped) bar chart using seaborn's 
    catplot. It represents two categorical variables by grouping bars of one 
    variable within levels of another. This visualization is particularly 
    effective for comparing a secondary categorical variable (the hue) within the 
    context of a primary categorical variable (the x-axis). The chart includes 
    automatic data labeling, text wrapping for long categorical labels, and 
    professional formatting for titles and captions.

    Clustered bar charts are essential for:
      * Epidemiology: Comparing disease prevalence across various age groups (primary) by gender (secondary).
      * Healthcare: Analyzing patient recovery outcomes across different treatment protocols (primary) by facility (secondary).
      * Intelligence Analysis: Visualizing intercepted signal counts across geographic regions (primary) by source type (secondary).
      * Data Science: Comparing model performance metrics across various datasets (primary) by algorithm type (secondary).
      * Public Health: Monitoring vaccine uptake across different administrative districts (primary) by age bracket (secondary).
      * Finance: Visualizing quarterly revenue across different product lines (primary) by market segment (secondary).
      * Operations: Comparing production defect rates across manufacturing lines (primary) by work shift (secondary).
      * Social Science: Analyzing survey response trends across demographic groups (primary) by political affiliation (secondary).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the categorical and numeric data to plot.
    grouping_column_name_1 : str
        The name of the primary categorical variable to be plotted on the x-axis.
    grouping_column_name_2 : str
        The name of the secondary categorical variable used for color encoding (hue).
    value_column_name : str
        The name of the column containing the numeric values to be represented by bar height.
    color_palette : str, optional
        The name of the seaborn color palette to use for the grouping levels. Defaults to "Set1".
    fill_transparency : float, optional
        The transparency level (alpha) of the bar fill, ranging from 0 to 1. Defaults to 0.8.
    display_order_list : list, optional
        A specific list of primary category names to define the horizontal order of groups. 
        If None, categories are sorted by value in descending order. Defaults to None.
    figure_size : tuple, optional
        A tuple of (width, height) in inches representing the desired plot dimensions. 
        Defaults to (6, 6).
    show_legend : bool, optional
        Whether to display the legend for the secondary categorical variable. Defaults to True.
    decimal_places_for_data_label : int, optional
        The number of decimal places to include in the labels at the top of each bar. 
        Defaults to 0.
    title_for_plot : str, optional
        The primary title text displayed at the top of the chart. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    x_indent : float, optional
        Horizontal offset for the titles and captions relative to the axes. Defaults to -0.04.
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
        The function displays the plot using matplotlib and optionally saves it to disk.

    Examples
    --------
    # Epidemiology: Infection rates by age group and gender
    import pandas as pd
    infection_df = pd.DataFrame({
        'Age Group': ['0-18', '19-40', '41-65', '65+'] * 2,
        'Gender': ['Male'] * 4 + ['Female'] * 4,
        'Incidence Rate': [45, 120, 85, 210, 40, 115, 80, 205]
    })
    PlotClusteredBarChart(
        infection_df, 'Age Group', 'Gender', 'Incidence Rate',
        title_for_plot="Influenza Incidence Analysis",
        subtitle_for_plot="Confirmed cases per 100,000 population stratified by age and gender"
    )

    # Healthcare: Patient wait times by department and priority
    wait_times_df = pd.DataFrame({
        'Department': ['Emergency', 'Cardiology', 'Oncology'] * 2,
        'Priority': ['High'] * 3 + ['Medium'] * 3,
        'Wait Time (min)': [25, 45, 30, 65, 80, 75]
    })
    PlotClusteredBarChart(
        wait_times_df, 'Department', 'Priority', 'Wait Time (min)',
        color_palette="muted",
        title_for_plot="Quarterly Wait Time Monitoring",
        subtitle_for_plot="Comparison of patient wait times by facility and urgency level"
    )

    # Intelligence Analysis: Intercept events by region and source
    intel_df = pd.DataFrame({
        'Region': ['East', 'West', 'North', 'South'] * 2,
        'Source': ['SIGINT'] * 4 + ['ELINT'] * 4,
        'Event Count': [120, 85, 210, 45, 80, 55, 150, 30]
    })
    PlotClusteredBarChart(
        intel_df, 'Region', 'Source', 'Event Count',
        color_palette="husl",
        title_for_plot="Signal Intercept Activity Summary",
        caption_for_plot="Regional monitoring analysis for mission cycle 2024-Q1."
    )
    """
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[grouping_column_name_1].unique())):
            raise ValueError("display_order_list must contain all of the categories in the " + grouping_column_name_1 + " column of dataframe.")
        else:
            # Order the dataframe by the display_order_list
            dataframe[grouping_column_name_1] = pd.Categorical(dataframe[grouping_column_name_1], categories=display_order_list, ordered=True)
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=True)[grouping_column_name_1].unique()
    
    # Draw a nested barplot by species and sex
    ax = sns.catplot(
        data=dataframe, 
        kind="bar",
        x=grouping_column_name_1, 
        y=value_column_name, 
        hue=grouping_column_name_2,
        palette=color_palette, 
        alpha=fill_transparency, 
        height=figure_size[1],
        aspect=figure_size[0] / figure_size[1],
        # ci=None,
        errorbar=None,
    )
    
    # Remove the legend if show_legend is False
    if show_legend == False:
        ax.ax.legend_.remove()
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Wrap x axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(grouping_column_name_1, 40, break_long_words=False))  # String wrap the variable name
    ax.ax.set_xlabel(wrapped_variable_name)
    
    # Format and wrap x axis tick labels using textwrap
    x_tick_labels = ax.ax.get_xticklabels()
    wrapped_x_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 20, break_long_words=False)) for label in x_tick_labels]
    ax.ax.set_xticklabels(
        wrapped_x_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Change x-axis colors
    ax.ax.tick_params(axis='x', colors="#262626")
    ax.ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.ax.spines['bottom'].set_visible(False)
    ax.ax.spines['right'].set_visible(False)
    ax.ax.spines['left'].set_visible(False)
    
    # Remove the y-axis tick text
    ax.ax.set_yticklabels([])
    
    # Add data labels to each bar in the plot
    # Get minimum value absolute value of the data
    min_value = dataframe[value_column_name].abs().min()
    for p in ax.ax.patches:
        # Get the height of each bar
        height = p.get_height()
        
        # Format the height to the specified number of decimal places
        height = round(height, decimal_places_for_data_label)
        
        # Add the data label to the plot, using the specified number of decimal places
        value_label = "{:,.{decimal_places}f}".format(height, decimal_places=decimal_places_for_data_label)
        if height != 0:
            ax.ax.text(
                x=p.get_x() + p.get_width() / 2.,
                y=height + (min_value * 0.05),
                s=value_label,
                ha="center",
                # fontname="Arial",
                fontsize=10,
                color="#262626"
            )
        
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    ax.ax.text(
        x=x_indent,
        y=title_y_indent,
        s=title_for_plot,
        # fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.ax.transAxes
    )
    
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        # fontname="Arial",
        fontsize=11,
        color="#666666",
        transform=ax.ax.transAxes
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
        ax.ax.text(
            x=x_indent,
            y=caption_y_indent,
            s=wrapped_caption,
            # fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.ax.transAxes
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

