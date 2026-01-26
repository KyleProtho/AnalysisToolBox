# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotBarChart(dataframe,
                 categorical_column_name, 
                 value_column_name,
                 # Bar formatting arguments
                 fill_color="#8eb3de",
                 color_palette=None,
                 top_n_to_highlight=None,
                 highlight_color="#b0170c",
                 fill_transparency=0.8,
                 # Plot formatting arguments
                 display_order_list=None,
                 figure_size=(8, 6),
                 # Text formatting arguments
                 title_for_plot=None,
                 subtitle_for_plot=None,
                 caption_for_plot=None,
                 data_source_for_plot=None,
                 title_y_indent=1.15,
                 subtitle_y_indent=1.1,
                 caption_y_indent=-0.15,
                 decimal_places_for_data_label=1,
                 data_label_fontsize=11,
                 data_label_padding=None,
                 # Plot saving arguments
                 filepath_to_save_plot=None):
    """
    Generate a formatted horizontal bar chart for categorical data comparison.

    This function creates a high-quality horizontal bar chart using the seaborn 
    library. It supports advanced features such as highlighting the top N 
    categories, custom color palettes, automatic text wrapping for long 
    categorical labels, and data sourcing annotations. The chart is designed 
    with a clean, professional aesthetic, featuring top-aligned x-axis labels 
    and customizable titles and subtitles.

    Horizontal bar charts are essential for:
      * Epidemiology: Comparing disease incidence rates across different age groups or districts.
      * Healthcare: Comparing the number of patient admissions across different medical departments.
      * Data Science: Comparing feature importance scores or model performance metrics.
      * Public Health: Visualizing the distribution of health resource allocations.
      * Marketing: Comparing customer preference scores for different product categories.
      * Finance: Visualizing budget allocations across various operational departments.
      * Social Science: Presenting survey response counts for multiple-choice questions.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the categorical and numeric data to plot.
    categorical_column_name : str
        The name of the column in the DataFrame representing the categories (y-axis).
    value_column_name : str
        The name of the column in the DataFrame representing the numeric values (x-axis).
    fill_color : str, optional
        A hex color code to use for all bars (e.g., "#8eb3de"). Used if `color_palette` 
        and `top_n_to_highlight` are not specified. Defaults to "#8eb3de".
    color_palette : str or list, optional
        The seaborn color palette name (e.g., "Set2") or a list of colors to use for 
        the bars. Defaults to None.
    top_n_to_highlight : int, optional
        The number of highest-value categories to highlight with `highlight_color`. 
        Other bars will be rendered in a neutral gray. Defaults to None.
    highlight_color : str, optional
        The hex color code used for highlighted bars (e.g., "#b0170c"). 
        Defaults to "#b0170c".
    fill_transparency : float, optional
        The transparency level (alpha) for the bar fill, ranging from 0 to 1. 
        Defaults to 0.8.
    display_order_list : list, optional
        A specific list of category names to define the vertical order of bars. 
        If None, bars are sorted by value in descending order. Defaults to None.
    figure_size : tuple, optional
        A tuple of (width, height) in inches for the figure. Defaults to (8, 6).
    title_for_plot : str, optional
        The primary title text displayed at the top of the chart. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the title. Defaults to None.
    caption_for_plot : str, optional
        Descriptive text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the axes. Defaults to 1.15.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the axes. Defaults to 1.1.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the axes. Defaults to -0.15.
    decimal_places_for_data_label : int, optional
        The number of decimal places to include in the numeric labels at the end 
        of each bar. Defaults to 1.
    data_label_fontsize : int, optional
        The font size for the numeric labels next to the bars. Defaults to 11.
    data_label_padding : float, optional
        The horizontal space between the end of the bar and the data label. 
        If None, it is calculated as 10% of the maximum value. Defaults to None.
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be saved. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the plot using matplotlib and optionally saves 
        it to disk.

    Examples
    --------
    # Epidemiology: Incidence of Influenza types
    import pandas as pd
    flu_data = pd.DataFrame({
        'Virus Strain': ['A(H1N1)', 'A(H3N2)', 'B/Victoria', 'B/Yamagata'],
        'Confirmed Cases': [1250, 840, 310, 120]
    })
    PlotBarChart(
        flu_data, 'Virus Strain', 'Confirmed Cases',
        top_n_to_highlight=1,
        title_for_plot="Influenza Strain Distribution",
        subtitle_for_plot="Total confirmed cases for the current flu season"
    )

    # Healthcare: Patient volume by department
    dept_df = pd.DataFrame({
        'Department': ['Emergency', 'Cardiology', 'Oncology', 'Pediatrics'],
        'Admissions': [450, 120, 85, 210]
    })
    PlotBarChart(
        dept_df, 'Department', 'Admissions',
        fill_color="#2ecc71",
        title_for_plot="Quarterly Patient Admissions",
        data_source_for_plot="Hospital EHR System"
    )
    """
    
    # Check that the column exists in the dataframe.
    if categorical_column_name not in dataframe.columns:
        raise ValueError("Column {} does not exist in dataframe.".format(categorical_column_name))
    
    # Make sure that the top_n_to_highlight is a positive integer, and less than the number of categories
    if top_n_to_highlight != None:
        if top_n_to_highlight < 0 or top_n_to_highlight > len(dataframe[categorical_column_name].value_counts()):
            raise ValueError("top_n_to_highlight must be a positive integer, and less than the number of categories.")
        
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[categorical_column_name].unique())):
            raise ValueError("display_order_list must contain all of the categories in the dataframe.")
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=False)[categorical_column_name]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)

    # Generate bar chart
    if top_n_to_highlight != None:
        ax = sns.barplot(
            data=dataframe,
            y=categorical_column_name,
            x=value_column_name,
            palette=[highlight_color if x in dataframe.sort_values(value_column_name, ascending=False)[value_column_name].nlargest(top_n_to_highlight).index else "#b8b8b8" for x in dataframe.sort_values(value_column_name, ascending=False).index],
            order=display_order_list,
            alpha=fill_transparency
        )
    elif color_palette != None:
        ax = sns.barplot(
            data=dataframe,
            y=categorical_column_name,
            x=value_column_name,
            hue=dataframe[categorical_column_name],
            palette=color_palette,
            order=display_order_list,
            alpha=fill_transparency
        )
    else:
        ax = sns.barplot(
            data=dataframe,
            y=categorical_column_name,
            x=value_column_name,
            color=fill_color,
            order=display_order_list,
            alpha=fill_transparency
        )
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Set the y axis ticks
    ax.set_yticks(range(len(dataframe[categorical_column_name].value_counts())))
    
    # Format and wrap y axis tick labels using textwrap
    y_tick_labels = ax.get_yticklabels()
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 40, break_long_words=False)) for label in y_tick_labels]
    ax.set_yticklabels(
        wrapped_y_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626"
    )
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
    # Change x-axis colors to "#666666"
    ax.tick_params(axis='x', colors="#666666")
    ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Get the maximum value of the x-axis
    if data_label_padding == None:
        max_x = dataframe[categorical_column_name].value_counts().max()
        data_label_padding = max_x * 0.10
    
    # Add data labels
    abs_values = dataframe.sort_values(value_column_name, ascending=False)[value_column_name].round(decimal_places_for_data_label)
    # Create format code for data labels
    data_label_format = "{:." + str(decimal_places_for_data_label) + "f}"
    lbls = [data_label_format.format(p) for p in abs_values]
    for i, p in enumerate(ax.patches):
        ax.text(
            p.get_width() + data_label_padding,
            p.get_y() + p.get_height() / 2,
            lbls[i],
            ha="left",
            va="center",
            fontsize=data_label_fontsize,
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

