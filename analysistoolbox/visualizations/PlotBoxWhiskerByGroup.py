# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotBoxWhiskerByGroup(dataframe,
                          value_column_name,
                          grouping_column_name_1,
                          grouping_column_name_2=None,
                          # Plot formatting arguments
                          fill_color="#8eb3de",
                          color_palette=None,
                          display_order_list=None,
                          show_legend=True,
                          # Text formatting arguments
                          title_for_plot=None,
                          subtitle_for_plot=None,
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          show_y_axis=False,
                          title_y_indent=1.1,
                          subtitle_y_indent=1.05,
                          caption_y_indent=-0.15,
                          x_indent=-0.10,
                          figure_size=(8, 6),
                          # Plot saving arguments
                          filepath_to_save_plot=None):
    """
    Create a formatted box-and-whisker plot for grouped data comparisons.

    This function generates a professional-grade box and whisker plot using seaborn 
    to visualize the distribution of a continuous variable across one or two 
    categorical grouping levels. It provides high-level control over aesthetics, 
    including custom color palettes, legend visibility, and automatic text 
    wrapping for categorical labels. The plot is designed to highlight central 
    tendency, dispersion, and potential outliers in an easy-to-read format.

    Box-and-whisker plots are essential for:
      * Epidemiology: Comparing the distribution of body mass index (BMI) across multiple demographic groups.
      * Healthcare: Analyzing the spread of patient recovery times between different treatment protocols.
      * Intelligence Analysis: Evaluating the distribution of response times for various threat detection systems.
      * Data Science: Detecting outliers and comparing feature distributions across target classes.
      * Public Health: Monitoring variability in air quality index levels across urban and rural zones.
      * Finance: Comparing the distribution of daily stock returns across different industry sectors.
      * Quality Control: Assessing the variance in product dimensions across different manufacturing shifts.
      * Social Science: Analyzing the distribution of standardized test scores across different school districts.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataset containing the numeric values and categorical 
        grouping variables.
    value_column_name : str
        The name of the continuous numeric variable to plot on the y-axis.
    grouping_column_name_1 : str
        The name of the primary categorical variable to plot on the x-axis.
    grouping_column_name_2 : str, optional
        The name of a secondary categorical variable used for nested grouping 
        and color encoding (hue). Defaults to None.
    fill_color : str, optional
        A hex color code used for the box interiors if `color_palette` is 
        not specified. Defaults to "#8eb3de".
    color_palette : str, optional
        The name of the seaborn color palette to use for grouping levels 
        (e.g., "Set2", "viridis"). Defaults to None.
    display_order_list : list, optional
        A specific list of categories to define the horizontal order of the 
        first grouping variable. If None, categories are sorted by the median 
        of the value column. Defaults to None.
    show_legend : bool, optional
        Whether to display the plot legend, particularly useful when 
        `grouping_column_name_2` is specified. Defaults to True.
    title_for_plot : str, optional
        The primary title text displayed at the top of the chart. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the origin of the data, appended to the caption. Defaults to None.
    show_y_axis : bool, optional
        Pass-through argument for consistency; the y-axis is managed by 
        seaborn's boxplot defaults. Defaults to False.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the axes. Defaults to 1.1.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the axes. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the axes. Defaults to -0.15.
    x_indent : float, optional
        Horizontal offset for the titles and captions relative to the axes. 
        Defaults to -0.10.
    figure_size : tuple, optional
        Dimensions of the figure as a (width, height) tuple in inches. Defaults to (8, 6).
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be 
        exported. If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the plot using matplotlib and optionally saves 
        it to disk.

    Examples
    --------
    # Epidemiology: Recovery days by treatment type and age group
    import pandas as pd
    epi_df = pd.DataFrame({
        'Recovery Days': [10, 12, 11, 14, 15, 13, 22, 25, 24, 30],
        'Treatment': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'Age Group': ['Adult', 'Child', 'Adult', 'Child', 'Adult', 'Child', 'Adult', 'Child', 'Adult', 'Child']
    })
    PlotBoxWhiskerByGroup(
        epi_df, 'Recovery Days', 'Treatment', 'Age Group',
        title_for_plot="Clinical Trial Recovery Analysis",
        subtitle_for_plot="Days to complete recovery stratified by treatment and patient age"
    )

    # Intelligence Analysis: Intercept latency by sensor type
    intel_df = pd.DataFrame({
        'Latency (ms)': [45, 52, 48, 120, 115, 118, 85, 92, 88],
        'Sensor': ['Alpha', 'Alpha', 'Alpha', 'Beta', 'Beta', 'Beta', 'Gamma', 'Gamma', 'Gamma']
    })
    PlotBoxWhiskerByGroup(
        intel_df, 'Latency (ms)', 'Sensor',
        fill_color="#2c3e50",
        title_for_plot="Sensor Array Performance",
        caption_for_plot="Latency measured during mission cycle 2024-05."
    )

    # Healthcare: Patient wait times by clinic location
    wait_df = pd.DataFrame({
        'Wait Time': [5, 10, 8, 45, 60, 55, 30, 25, 35],
        'Location': ['Downtown', 'Downtown', 'Downtown', 'Suburb', 'Suburb', 'Suburb', 'East', 'East', 'East']
    })
    PlotBoxWhiskerByGroup(
        wait_df, 'Wait Time', 'Location',
        color_palette="muted",
        title_for_plot="Patient Access Monitoring",
        subtitle_for_plot="Distribution of wait times across satellite clinics",
        show_legend=False
    )
    """
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[grouping_column_name_1].unique())):
            raise ValueError("display_order_list must contain all of the categories in the " + grouping_column_name_1 + " column of dataframe.")
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=True)[grouping_column_name_1].unique()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Generate box whisker plot
    if grouping_column_name_2 != None:
        if fill_color != None:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                hue=grouping_column_name_2, 
                color=fill_color,
                order=display_order_list,
            )
        else:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                hue=grouping_column_name_2, 
                palette=color_palette,
                order=display_order_list,
            )
    else:
        if fill_color != None:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                color=fill_color,
                order=display_order_list,
            )
        else:
            ax = sns.boxplot(
                data=dataframe,
                x=grouping_column_name_1,
                y=value_column_name, 
                palette=color_palette,
                order=display_order_list,
            )
    
    # Remove top, and right spines. Set bottom and left spine to dark gray.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#262626")
    ax.spines['left'].set_color("#262626")
    
    # If show_legend is False, remove the legend
    if show_legend == False:
        ax.legend_.remove()
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
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
    
    # String wrap group variable 1 tick labels
    group_variable_1_tick_labels = ax.get_xticklabels()
    group_variable_1_tick_labels = [label.get_text() for label in group_variable_1_tick_labels]
    group_variable_1_tick_labels = ['\n'.join(textwrap.wrap(label, 20, break_long_words=False)) for label in group_variable_1_tick_labels]
    ax.set_xticklabels(group_variable_1_tick_labels)
    
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
    
    # Set y-axis tick label font to Arial, size 9, and color #666666
    ax.tick_params(
        axis='y',
        which='major',
        labelsize=9,
        labelcolor="#666666",
        pad=2,
        bottom=True,
        labelbottom=True
    )
    # plt.yticks(fontname='Arial')
    
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

