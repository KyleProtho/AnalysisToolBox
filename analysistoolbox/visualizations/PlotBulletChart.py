# Load packages
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotBulletChart(dataframe, 
                    value_column_name, 
                    grouping_column_name,
                    target_value_column_name=None,
                    list_of_limit_columns=None,
                    value_maximum=None,
                    value_minimum=0,
                    # Bullet chart formatting arguments
                    display_order_list=None,
                    null_background_color='#dbdbdb',
                    background_color_palette=None,
                    background_alpha=0.8,
                    value_dot_size=200,
                    value_dot_color="#383838",
                    value_dot_color_by_level=False,
                    value_dot_outline_color="#ffffff",
                    value_dot_outline_width=2,
                    target_line_color='#bf3228',
                    target_line_width=4,
                    figure_size=(8, 6),
                    # Text formatting arguments
                    show_value_labels=True,
                    value_label_color="#262626",
                    value_label_format="{:.0f}",
                    value_label_font_size=12,
                    value_label_spacing=.65,
                    value_label_padding=0.3,
                    value_label_background_color="#ffffff",
                    value_label_background_alpha=0.85,
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
    Generate a formatted horizontal bullet chart to visualize performance against targets and ranges.

    This function creates a high-quality horizontal bullet chart, which is a variation 
    of a bar chart designed to replace dashboard gauges and meters. It displays a 
    single primary measure (represented by a dot), compares it to a target value 
    (represented by a vertical line), and orients it within qualitative ranges of 
    performance (e.g., poor, satisfactory, good) shown as background shades. The 
    function provides extensive customization for colors, labels, and plot annotations.

    Bullet charts are essential for:
      * Healthcare: Monitoring surgeon success rates against departmental benchmarks and thresholds.
      * Epidemiology: Tracking actual vaccination rates vs. herd immunity targets across regions.
      * Project Management: Monitoring task completion percentages vs. project milestones.
      * Public Health: Comparing current emergency room wait times against historical performance bands.
      * Data Science: Evaluating model accuracy metrics against baseline and state-of-the-art targets.
      * Operations: Monitoring production efficiency across several manufacturing lines.
      * Finance: Visualizing actual expenditure vs. budget targets across multiple departments.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the performance data to be plotted.
    value_column_name : str
        The name of the column containing the primary measure (performance value).
    grouping_column_name : str
        The name of the column representing the categories or groups for each bullet.
    target_value_column_name : str, optional
        The name of the column containing the target values for each group. Defaults to None.
    list_of_limit_columns : list of str, optional
        A list of column names containing the qualitative range boundaries (e.g., ['poor', 'fair', 'good']). 
        Defaults to None.
    value_maximum : float, optional
        The absolute maximum value for the chart's x-axis. If None, it is inferred from 
        the data. Defaults to None.
    value_minimum : float, optional
        The absolute minimum value for the chart's x-axis. Defaults to 0.
    display_order_list : list of str, optional
        A specific list of category names to define the vertical order of the bullets. 
        If None, categories are sorted by their `value` column. Defaults to None.
    null_background_color : str, optional
        The hex color code for the background area where no qualitative ranges are defined. 
        Defaults to '#dbdbdb'.
    background_color_palette : str, optional
        The name of the seaborn color palette used for qualitative ranges. If None, 
        a professional greyscale palette is used. Defaults to None.
    background_alpha : float, optional
        The transparency level (alpha) of the qualitative range background colors. 
        Defaults to 0.8.
    value_dot_size : int, optional
        The area of the marker representing the primary measure. Defaults to 200.
    value_dot_color : str, optional
        The hex color code for the value marker. Defaults to "#383838".
    value_dot_color_by_level : bool, optional
        Whether to color the value marker based on which qualitative range it falls into. 
        Defaults to False.
    value_dot_outline_color : str, optional
        The hex color code for the border of the value marker. Defaults to "#ffffff".
    value_dot_outline_width : int, optional
        The width of the border around the value marker. Defaults to 2.
    target_line_color : str, optional
        The hex color code for the target value marker line. Defaults to '#bf3228'.
    target_line_width : int, optional
        The width of the vertical target line. Defaults to 4.
    figure_size : tuple of int, optional
        Dimensions of the output figure as a (width, height) tuple in inches. Defaults to (8, 6).
    show_value_labels : bool, optional
        Whether to display numeric text labels next to each value marker. Defaults to True.
    value_label_color : str, optional
        The hex color code for the numeric text labels. Defaults to "#262626".
    value_label_format : str, optional
        The Python string format for the numeric labels (e.g., "{:.1f}%"). Defaults to "{:.0f}".
    value_label_font_size : int, optional
        The font size for the numeric text labels. Defaults to 12.
    value_label_spacing : float, optional
        Vertical offset for positioning the numeric label relative to the marker. 
        Defaults to 0.65.
    value_label_padding : float, optional
        Padding inside the background box of the numeric label. Defaults to 0.3.
    value_label_background_color : str, optional
        The hex color code for the background box behind the numeric label. Defaults to "#ffffff".
    value_label_background_alpha : float, optional
        The transparency of the numeric label's background box. Defaults to 0.85.
    title_for_plot : str, optional
        The primary title text displayed at the top of the chart. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
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
    # Healthcare: Surgeon success rates vs. target and benchmarks
    import pandas as pd
    surgeon_df = pd.DataFrame({
        'Surgeon': ['Dr. Smith', 'Dr. Jones', 'Dr. Brown'],
        'Success Rate': [92, 88, 95],
        'Target': [90, 90, 90],
        'Poor': [70, 70, 70],
        'Good': [85, 85, 85],
        'Excellent': [95, 95, 95]
    })
    PlotBulletChart(
        surgeon_df, 'Success Rate', 'Surgeon', 'Target',
        list_of_limit_columns=['Poor', 'Good', 'Excellent'],
        title_for_plot="Cardiac Surgery Success Rates",
        subtitle_for_plot="Individual surgeon performance against hospital standards"
    )

    # Epidemiology: Vaccination Progress by Region
    vax_df = pd.DataFrame({
        'Region': ['Capitol City', 'Everglades', 'Highlands'],
        'Vax Rate': [65, 42, 78],
        'Goal': [80, 80, 80],
        'Threshold': [60, 60, 60]
    })
    PlotBulletChart(
        vax_df, 'Vax Rate', 'Region', 'Goal',
        list_of_limit_columns=['Threshold'],
        background_color_palette="RdYlGn",
        title_for_plot="COVID-19 Vaccination Progress",
        value_label_format="{:.1f}%"
    )
    """
    
    # Ensure that the number of unique values in the group column is equal to the number of rows in the dataframe
    if len(dataframe[grouping_column_name].unique()) != len(dataframe):
        raise ValueError("The number of unique values in the group column must be equal to the number of rows in the dataframe.")
    
    # If value_maximum is not specified, use the maximum value among the limit columns
    if value_maximum is None:
        if list_of_limit_columns is not None:
            value_maximum = dataframe[list_of_limit_columns].max().max()
        else:
            value_maximum = dataframe[value_column_name].max()
    
    # If display_order_list is provided, check that it contains all of the categories in the dataframe
    if display_order_list != None:
        if not set(display_order_list).issubset(set(dataframe[grouping_column_name].unique())):
            raise ValueError("display_order_list must contain all of the categories in the " + grouping_column_name + " column of dataframe.")
        else:
            # Reverse the display_order_list
            display_order_list = display_order_list[::-1]
            # Order the dataframe by the display_order_list
            dataframe[grouping_column_name] = pd.Categorical(dataframe[grouping_column_name], categories=display_order_list, ordered=True)
            dataframe = dataframe.sort_values(grouping_column_name)
    else:
        # If display_order_list is not provided, create one from the dataframe
        display_order_list = dataframe.sort_values(value_column_name, ascending=True)[grouping_column_name].unique()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    # Iterate over each row in the dataframe
    for index, row in dataframe.iterrows():
        group = row[grouping_column_name]
        value = row[value_column_name]
        if target_value_column_name != None:
            target = row[target_value_column_name]
            
        # Plot the background area color
        bars = ax.barh(
            y=group, 
            width=value_maximum-value_minimum, 
            color=null_background_color, 
            height=0.8
        )
        
        # Get the y-position of the background area color
        y_position = bars[0].get_y()
        
        # If limit columns are specified, plot the limit columns
        if list_of_limit_columns != None:
            # Get the colors from the specified color palette, using the number of limit columns
            if background_color_palette == None or value_dot_color_by_level == True:
                # Create a greyscale palette
                colors = sns.light_palette("#acacad", len(list_of_limit_columns) + 1)
                # Reverse the palette
                colors = colors[::-1]
            else:
                colors = sns.color_palette(background_color_palette, len(list_of_limit_columns) + 1)

            # Rearrange the limit columns in descending order according to each of their values
            sorted_row = row[list_of_limit_columns].sort_values(ascending=False)
            
            # Iterate over each limit column
            for i in range(len(list_of_limit_columns)):
                # Plot the limit column value as a bar
                sorted_row_value = sorted_row[i]
                if i == 0 and np.isnan(sorted_row_value) == False:
                    ax.barh(
                        y=group, 
                        width=value_maximum-value_minimum, 
                        color=colors[i], 
                        height=0.8
                    )
                # If the limit column value is not null plot the limit column value as a bar:
                if np.isnan(sorted_row_value) == False:
                    ax.barh(
                        y=group, 
                        width=sorted_row_value-value_minimum, 
                        color=colors[i+1], 
                        height=0.8,
                        edgecolor='white',
                        linewidth=2,
                        alpha=background_alpha,
                    )
                else:
                    continue
        
        # Plot the target value as a line in the group
        if target_value_column_name != None:
            ax.plot(
                [target, target], 
                [y_position+0.2, y_position+0.6], 
                color=target_line_color, 
                linewidth=target_line_width
            )

        # Plot value as a large dot
        if value_dot_color_by_level==False:
            ax.scatter(
                value, 
                group, 
                color=value_dot_color, 
                s=value_dot_size,
                edgecolor=value_dot_outline_color,
                linewidth=value_dot_outline_width
            )
        else:
            # Get the colors from the specified color palette, using the number of limit columns
            colors = sns.color_palette(background_color_palette, len(list_of_limit_columns) + 1)
            
            # Iterate through each row in the dataframe
            for index, row in dataframe.iterrows():
                # Get the value of the limit columns
                limit_values = row[list_of_limit_columns]
                
                # Get the colors from the specified color palette, using the number of limit columns
                colors = sns.color_palette(background_color_palette, len(list_of_limit_columns) + 1)
                
                # Assign the color of the dot based on the value of the limit columns
                for i in range(len(limit_values)):
                    if value <= limit_values[i]:
                        value_dot_color = colors[i+1]
                        break
                    else:
                        value_dot_color = "#383838"
                
                # Plot value as a large dot
                ax.scatter(
                    value, 
                    group, 
                    color=value_dot_color, 
                    s=value_dot_size,
                    edgecolor=value_dot_outline_color,
                    linewidth=value_dot_outline_width
                )
        
        # Add data label for the value, if requested
        if show_value_labels:
            ax.text(
                x=value,
                y=y_position+value_label_spacing,
                s=value_label_format.format(value),
                # fontname="Arial",
                fontsize=value_label_font_size,
                color=value_label_color,
                verticalalignment="center",
                horizontalalignment="center",
                # Add a white background to the data label
                bbox=dict(
                    facecolor=value_label_background_color, 
                    alpha=value_label_background_alpha, 
                    edgecolor='none',
                    boxstyle="round,pad="+str(value_label_padding)
                )
            )
        
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
    # Format and wrap y axis tick labels using textwrap
    wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 30, break_long_words=False)) for label in ax.get_yticklabels()]
    ax.set_yticklabels(
        wrapped_y_tick_labels, 
        fontsize=10, 
        # fontname="Arial", 
        color="#262626",
    )
    
    # Set the x indent of the plot titles and captions
    # Get longest y tick label
    longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
    if len(longest_y_tick_label) >= 30:
        x_indent = -0.3
    else:
        x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)
    
    # Change x-axis colors to "#666666"
    ax.tick_params(axis='x', colors="#666666")
    ax.spines['top'].set_color("#666666")
    
    # Remove bottom, left, and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Move x-axis to the top
    ax.xaxis.tick_top()
    
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

