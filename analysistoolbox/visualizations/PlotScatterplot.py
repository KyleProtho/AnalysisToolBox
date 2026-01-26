# Load packages
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import textwrap

# Declare function
def PlotScatterplot(dataframe,
                    y_axis_column_name,
                    x_axis_column_name,
                    # Dot formatting arguments
                    grouping_column_name=None,
                    group_color_palette="Set1",
                    dot_fill_color="#999999",
                    size_by_column_name=None,
                    size_normalization=None,
                    # Fitted line arguments
                    fitted_line_type=None,
                    line_color=None,
                    # Text formatting arguments
                    title_for_plot=None,
                    subtitle_for_plot=None,
                    caption_for_plot=None,
                    data_source_for_plot=None,
                    x_indent=-0.128,
                    title_y_indent=1.125,
                    subtitle_y_indent=1.05,
                    caption_y_indent=-0.3,
                    # Quadrant arguments
                    upper_left_quadrant_label=None,
                    upper_left_quadrant_fill_color=None,
                    upper_right_quadrant_label=None,
                    upper_right_quadrant_fill_color=None,
                    lower_left_quadrant_label=None,
                    lower_left_quadrant_fill_color=None,
                    lower_right_quadrant_label=None,
                    lower_right_quadrant_fill_color=None,
                    # Plot saving arguments
                    filepath_to_save_plot=None):
    """
    Generate a formatted scatterplot with optional regression lines and quadrants.

    This function creates a high-quality scatterplot using seaborn, allowing for 
    the visualization of the relationship between two continuous numeric variables. 
    It supports grouping by categorical variables (using color), sizing points 
    by a third numeric variable, and fitting various regression lines (straight 
    linear or LOWESS). Additionally, the plot can be divided into four labeled 
    quadrants with custom background colors, which is useful for categorization 
    and performance mapping.

    Scatterplots are essential for:
      * Epidemiology: Examining the relationship between population density and transmission rates.
      * Healthcare: Correlating patient age with physiological metrics like blood pressure or heart rate.
      * Intelligence Analysis: Analyzing the relationship between insurgent activity and local economic stability.
      * Data Science: Identifying patterns, clusters, and multi-collinearity during feature engineering.
      * Public Health: Correlating socioeconomic status indices with local life expectancy.
      * Finance: Visualizing the risk (standard deviation) vs. return of various asset classes.
      * Marketing: Examining the correlation between advertising spend and customer conversion volume.
      * Quality Control: Monitoring the relationship between manufacturing temperature and defect counts.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the numeric and categorical data to plot.
    y_axis_column_name : str
        The name of the column to be plotted on the vertical y-axis.
    x_axis_column_name : str
        The name of the column to be plotted on the horizontal x-axis.
    grouping_column_name : str, optional
        The name of a categorical column used for color-encoding (hue). Defaults to None.
    group_color_palette : str, optional
        The name of the seaborn color palette to use for groupings. Defaults to "Set1".
    dot_fill_color : str, optional
        The hex color code for points when no grouping is applied. Defaults to "#999999".
    size_by_column_name : str, optional
        The name of a numeric column used to scale the size of the scatter points. 
        Defaults to None.
    size_normalization : tuple, optional
        A (min, max) tuple defining the range of point sizes. Defaults to None.
    fitted_line_type : {None, 'straight', 'lowess'}, optional
        The type of regression line to overlay on the plot. 'straight' fits a 
        linear regression, while 'lowess' fits a smooth non-linear curve. 
        Defaults to None.
    line_color : str, optional
        The hex color code for the fitted regression line. If None, it matches 
        the `dot_fill_color`. Defaults to None.
    title_for_plot : str, optional
        The primary title text at the top of the chart. Defaults to None.
    subtitle_for_plot : str, optional
        Descriptive subtitle text below the main title. Defaults to None.
    caption_for_plot : str, optional
        Explanatory text or notes at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    x_indent : float, optional
        Horizontal offset for the titles and captions relative to the axes. 
        Defaults to -0.128.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the grid. Defaults to 1.125.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the grid. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the grid. Defaults to -0.3.
    upper_left_quadrant_label : str, optional
        Text label to display in the upper-left quadrant area. Defaults to None.
    upper_left_quadrant_fill_color : str, optional
        Hex color code for the upper-left quadrant background. Defaults to None.
    upper_right_quadrant_label : str, optional
        Text label to display in the upper-right quadrant area. Defaults to None.
    upper_right_quadrant_fill_color : str, optional
        Hex color code for the upper-right quadrant background. Defaults to None.
    lower_left_quadrant_label : str, optional
        Text label to display in the lower-left quadrant area. Defaults to None.
    lower_left_quadrant_fill_color : str, optional
        Hex color code for the lower-left quadrant background. Defaults to None.
    lower_right_quadrant_label : str, optional
        Text label to display in the lower-right quadrant area. Defaults to None.
    lower_right_quadrant_fill_color : str, optional
        Hex color code for the lower-right quadrant background. Defaults to None.
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the scatterplot using matplotlib and optionally 
        saves it to disk.

    Examples
    --------
    # Epidemiology: Correlation between age and symptom severity
    import pandas as pd
    import numpy as np
    epi_df = pd.DataFrame({
        'Age': np.random.randint(5, 85, 100),
        'Severity Score': np.random.uniform(0, 100, 100),
        'Condition': np.random.choice(['A', 'B'], 100)
    })
    PlotScatterplot(
        epi_df, 'Severity Score', 'Age', grouping_column_name='Condition',
        fitted_line_type='lowess',
        title_for_plot="Age-Severity Analysis by Condition",
        subtitle_for_plot="Non-linear relationship between patient age and reported symptom scores"
    )

    # Intelligence Analysis: Insurgent activity vs. economic stability
    intel_df = pd.DataFrame({
        'Stability Index': np.random.uniform(0, 1, 50),
        'Incident Count': np.random.randint(0, 50, 50)
    })
    PlotScatterplot(
        intel_df, 'Incident Count', 'Stability Index',
        upper_left_quadrant_label="High Risk / Fragile Economy",
        upper_left_quadrant_fill_color="#e74c3c",
        lower_right_quadrant_label="Stable / Minimal Activity",
        lower_right_quadrant_fill_color="#2ecc71",
        title_for_plot="Regional Security Assessment",
        subtitle_for_plot="Correlation between sector economic indicators and tactical incidents"
    )

    # Healthcare: Health metrics correlation
    hosp_df = pd.DataFrame({
        'Daily Steps': np.random.randint(1000, 15000, 80),
        'BMI': np.random.uniform(18, 35, 80)
    })
    PlotScatterplot(
        hosp_df, 'BMI', 'Daily Steps',
        fitted_line_type='straight',
        line_color="#2c3e50",
        title_for_plot="Patient Wellness Monitoring",
        caption_for_plot="Data collected through hospital-provided wearable sensors."
    )
    """
    
    # Ensure that fitted_line_type is None, 'straight', or 'lowess'
    if fitted_line_type not in [None, 'straight', 'lowess']:
        raise ValueError("Invalid fitted_line_type argument. Please enter None, 'straight', or 'lowess'.")
    
    # If line color is not specified, use fill color
    if line_color == None:
        line_color = dot_fill_color
        
    # Create list of quadrant labels
    quadrant_labels = [upper_left_quadrant_label, upper_right_quadrant_label, lower_left_quadrant_label, lower_right_quadrant_label]
    
    # Draw scatterplot
    if fitted_line_type == None:
        if grouping_column_name == None:
            ax = sns.scatterplot(
                data=dataframe, 
                x=x_axis_column_name,
                y=y_axis_column_name,
                size=size_by_column_name,
                size_norm=size_normalization,
                color=dot_fill_color,
                alpha=0.5,
                linewidth=0.5,
                edgecolor=dot_fill_color
            )
        else:
            ax = sns.scatterplot(
                data=dataframe, 
                x=x_axis_column_name,
                y=y_axis_column_name,
                size=size_by_column_name,
                size_norm=size_normalization,
                hue=grouping_column_name,
                alpha=0.5,
                linewidth=0.5,
                palette=group_color_palette
            )
    elif fitted_line_type == 'straight':
        if grouping_column_name == None:
            ax = sns.regplot(
                data=dataframe,
                x=x_axis_column_name,
                y=y_axis_column_name,
                marker='o',
                scatter_kws={
                    'color': dot_fill_color,
                    'alpha': 0.5,
                    'linewidths': 0.5,
                    'edgecolor': dot_fill_color,
                    's': dataframe[size_by_column_name] * size_normalization[1] if size_by_column_name and size_normalization else (dataframe[size_by_column_name] if size_by_column_name else 20),
                },
                fit_reg=True,
                line_kws={'color': line_color}
            )
        else:
            ax = sns.lmplot(
                data=dataframe,
                x=x_axis_column_name,
                y=y_axis_column_name,
                hue=grouping_column_name,
                palette=group_color_palette,
                scatter_kws={
                    'alpha': 0.5,
                    'linewidths': 0.5,
                    's': dataframe[size_by_column_name] * size_normalization[1] if size_by_column_name and size_normalization else (dataframe[size_by_column_name] if size_by_column_name else 20),
                },
                fit_reg=True
            )
    elif fitted_line_type == 'lowess':
        if grouping_column_name == None:
            ax = sns.regplot(
                data=dataframe,
                x=x_axis_column_name,
                y=y_axis_column_name,
                marker='o',
                scatter_kws={
                    'color': dot_fill_color,
                    'alpha': 0.5,
                    'linewidths': 0.5,
                    'edgecolor': dot_fill_color,
                    's': dataframe[size_by_column_name] * size_normalization[1] if size_by_column_name and size_normalization else (dataframe[size_by_column_name] if size_by_column_name else 20),
                },
                lowess=True,
                line_kws={'color': line_color}
            )
        else:
            ax = sns.lmplot(
                data=dataframe,
                x=x_axis_column_name,
                y=y_axis_column_name,
                hue=grouping_column_name,
                palette=group_color_palette,
                scatter_kws={
                    'alpha': 0.5,
                    'linewidths': 0.5,
                    's': dataframe[size_by_column_name] * size_normalization[1] if size_by_column_name and size_normalization else (dataframe[size_by_column_name] if size_by_column_name else 20),
                },
                lowess=True
            )
        
    # Remove top and right spines, and set bottom and left spines to gray
    try:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
    except AttributeError:
        ax.ax.spines['top'].set_visible(False)
        ax.ax.spines['right'].set_visible(False)
        ax.ax.spines['bottom'].set_color('#666666')
        ax.ax.spines['left'].set_color('#666666')
    
    # Format tick labels to be Arial, size 9, and color #666666
    try:
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
        )
    except AttributeError:
        ax.ax.tick_params(
            which='major', 
            labelsize=9, 
            color='#666666'
        )
    
    # If any of the quadrant labels are not None, find the middle of the x and y axes
    if any(quadrant_labels):
        try:
            x_start = ax.get_xlim()[0]
            x_end = ax.get_xlim()[1]
            y_start = ax.get_ylim()[0]
            y_end = ax.get_ylim()[1]
        except AttributeError:
            x_start = ax.ax.get_xlim()[0]
            x_end = ax.ax.get_xlim()[1]
            y_start = ax.ax.get_ylim()[0]
            y_end = ax.ax.get_ylim()[1]
        x_middle = x_start + (x_end - x_start) / 2
        y_middle = y_start + (y_end - y_start) / 2
        
        # Add upper left quadrant
        if upper_left_quadrant_label != None:
            # Get coordinates for upper left quadrant label
            upper_left_quadrant_label_x = x_start + (x_middle - x_start) / 2
            upper_left_quadrant_label_x = upper_left_quadrant_label_x + (x_start - upper_left_quadrant_label_x) / 2
            upper_left_quadrant_label_y = y_middle + (y_end - y_middle) / 2
            upper_left_quadrant_label_y = upper_left_quadrant_label_y + (y_end - upper_left_quadrant_label_y) / 2
            # If no fill color is specified, use dot fill color
            if upper_left_quadrant_fill_color == None:
                upper_left_quadrant_fill_color = dot_fill_color
            # Write text in upper left quadrant
            try:
                ax.text(
                    upper_left_quadrant_label_x, 
                    upper_left_quadrant_label_y, 
                    upper_left_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=upper_left_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            except AttributeError:
                ax.ax.text(
                    upper_left_quadrant_label_x, 
                    upper_left_quadrant_label_y, 
                    upper_left_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=upper_left_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            # Add a rectangle in the upper left quadrant, behind the dot plot
            if fitted_line_type == None:
                try:
                    ax.add_patch(
                        Rectangle(
                            (x_start, y_middle),
                            x_middle - x_start,
                            y_end - y_middle,
                            edgecolor='none',
                            facecolor=upper_left_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
                except AttributeError:
                    ax.ax.add_patch(
                        Rectangle(
                            (x_start, y_middle),
                            x_middle - x_start,
                            y_end - y_middle,
                            edgecolor='none',
                            facecolor=upper_left_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
        
        # Add upper right quadrant
        if upper_right_quadrant_label != None:
            # Get coordinates for upper right quadrant label
            upper_right_quadrant_label_x = x_middle + (x_end - x_middle) / 2
            upper_right_quadrant_label_x = upper_right_quadrant_label_x + (x_end - upper_right_quadrant_label_x) / 2
            upper_right_quadrant_label_y = y_middle + (y_end - y_middle) / 2
            upper_right_quadrant_label_y = upper_right_quadrant_label_y + (y_end - upper_right_quadrant_label_y) / 2
            # If no fill color is specified, use dot fill color
            if upper_right_quadrant_fill_color == None:
                upper_right_quadrant_fill_color = dot_fill_color
            # Write text in upper right quadrant
            try:
                ax.text(
                    upper_right_quadrant_label_x, 
                    upper_right_quadrant_label_y, 
                    upper_right_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=upper_right_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            except AttributeError:
                ax.ax.text(
                    upper_right_quadrant_label_x, 
                    upper_right_quadrant_label_y, 
                    upper_right_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=upper_right_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            # Add a rectangle in the upper right quadrant, behind the dot plot
            if fitted_line_type == None:
                try:
                    ax.add_patch(
                        Rectangle(
                            (x_middle, y_middle),
                            x_end - x_middle,
                            y_end - y_middle,
                            edgecolor='none',
                            facecolor=upper_right_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
                except AttributeError:
                    ax.ax.add_patch(
                        Rectangle(
                            (x_middle, y_end),
                            x_middle - x_end,
                            y_end - y_middle,
                            edgecolor='none',
                            facecolor=upper_right_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
                
        # Add lower left quadrant
        if lower_left_quadrant_label != None:
            # Get coordinates for lower left quadrant label
            lower_left_quadrant_label_x = x_start + (x_middle - x_start) / 2
            lower_left_quadrant_label_x = lower_left_quadrant_label_x + (x_start - lower_left_quadrant_label_x) / 2
            lower_left_quadrant_label_y = y_start + (y_middle - y_start) / 2
            lower_left_quadrant_label_y = lower_left_quadrant_label_y - (y_middle - lower_left_quadrant_label_y) / 2
            # If no fill color is specified, use dot fill color
            if lower_left_quadrant_fill_color == None:
                lower_left_quadrant_fill_color = dot_fill_color
            # Write text in lower left quadrant
            try:
                ax.text(
                    lower_left_quadrant_label_x, 
                    lower_left_quadrant_label_y, 
                    lower_left_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=lower_left_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            except AttributeError:
                ax.ax.text(
                    lower_left_quadrant_label_x, 
                    lower_left_quadrant_label_y, 
                    lower_left_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=lower_left_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            # Add a rectangle in the lower left quadrant, behind the dot plot
            if fitted_line_type == None:
                try:
                    ax.add_patch(
                        Rectangle(
                            (x_start, y_start),
                            x_middle - x_start,
                            y_middle - y_start,
                            edgecolor='none',
                            facecolor=lower_left_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
                except AttributeError:
                    ax.ax.add_patch(
                        Rectangle(
                            (x_start, y_start),
                            x_middle - x_start,
                            y_middle - y_start,
                            edgecolor='none',
                            facecolor=lower_left_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
                
        # Add lower right quadrant
        if lower_right_quadrant_label != None:
            # Get coordinates for lower right quadrant label
            lower_right_quadrant_label_x = x_middle + (x_end - x_middle) / 2
            lower_right_quadrant_label_x = lower_right_quadrant_label_x + (x_end - lower_right_quadrant_label_x) / 2
            lower_right_quadrant_label_y = y_start + (y_middle - y_start) / 2
            lower_right_quadrant_label_y = lower_right_quadrant_label_y - (y_middle - lower_right_quadrant_label_y) / 2
            # If no fill color is specified, use dot fill color
            if lower_right_quadrant_fill_color == None:
                lower_right_quadrant_fill_color = dot_fill_color
            # Write text in lower right quadrant
            try:
                ax.text(
                    lower_right_quadrant_label_x, 
                    lower_right_quadrant_label_y, 
                    lower_right_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=lower_right_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            except AttributeError:
                ax.ax.text(
                    lower_right_quadrant_label_x, 
                    lower_right_quadrant_label_y, 
                    lower_right_quadrant_label, 
                    fontsize=10,
                    fontweight='bold',
                    color=lower_right_quadrant_fill_color,
                    alpha=0.8,
                    ha='center', 
                    va='center'
                )
            # Add a rectangle in the lower right quadrant, behind the dot plot
            if fitted_line_type == None:
                try:
                    ax.add_patch(
                        Rectangle(
                            (x_middle, y_start),
                            x_end - x_middle,
                            y_middle - y_start,
                            edgecolor='none',
                            facecolor=lower_right_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
                except AttributeError:
                    ax.ax.add_patch(
                        Rectangle(
                            (x_middle, y_start),
                            x_end - x_middle,
                            y_middle - y_start,
                            edgecolor='none',
                            facecolor=lower_right_quadrant_fill_color,
                            alpha=0.15
                        )
                    ).set_zorder(-1)
        
        # Insert a vertical line at the middle of the x axis
        plt.axvline(x_middle, color='#999999', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Insert a horizontal line at the middle of the y axis
        plt.axhline(y_middle, color='#999999', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    try:
        ax.text(
            x=x_indent,
            y=title_y_indent,
            s=title_for_plot,
            # fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
    except AttributeError:
        ax.ax.text(
            x=x_indent,
            y=title_y_indent-0.05,
            s=title_for_plot,
            # fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.ax.transAxes
        )
        
    # Set the subtitle with Arial font, size 11, and color #666666
    try:
        ax.text(
            x=x_indent,
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            # fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
    except AttributeError:
        ax.ax.text(
            x=x_indent,
            y=subtitle_y_indent-0.023,
            s=subtitle_for_plot,
            # fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.ax.transAxes
        )
        
    # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
    try:
        ax.yaxis.set_label_coords(-0.1, 0.99)
        ax.yaxis.set_label_text(
            textwrap.fill(y_axis_column_name, 30, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right'
        )
    except AttributeError:
        ax.ax.yaxis.set_label_coords(-0.1, 0.99)
        ax.ax.yaxis.set_label_text(
            textwrap.fill(y_axis_column_name, 30, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right'
        )
        
    # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
    try:
        ax.xaxis.set_label_coords(0.99, -0.1)
        ax.xaxis.set_label_text(
            textwrap.fill(x_axis_column_name, 30, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right'
        )
    except AttributeError:
        ax.ax.xaxis.set_label_coords(0.99, -0.075)
        ax.ax.xaxis.set_label_text(
            textwrap.fill(x_axis_column_name, 30, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right'
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
        try:
            ax.text(
                x=x_indent,
                y=caption_y_indent,
                s=wrapped_caption,
                # fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
        except AttributeError:
            ax.ax.text(
                x=x_indent,
                y=caption_y_indent+0.075,
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

