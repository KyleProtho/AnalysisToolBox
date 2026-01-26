# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotContingencyHeatmap(dataframe,
                           categorical_column_name_1,
                           categorical_column_name_2,
                           normalize_by="columns",
                           # Plot formatting arguments
                           color_palette="Blues",
                           show_legend=False,
                           figure_size=(8, 6),
                           # Text formatting arguments
                           data_label_format=".1%",
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
    Generate a formatted contingency heatmap to visualize relationships between categorical variables.

    This function creates a high-quality heatmap based on a contingency table 
    (cross-tabulation) of two categorical variables. It supports normalization 
    by rows, columns, or the entire dataset to show relative frequencies. The 
    visualization is designed for professional reports, featuring automatic text 
    wrapping for long category names, customizable color scales, and 
    annotated data labels in various formats (e.g., percentages or counts).

    Contingency heatmaps are essential for:
      * Epidemiology: Analyzing the relationship between exposure levels and health outcomes.
      * Healthcare: Visualizing patient triage categories across different hospital departments.
      * Data Science: Evaluating the confusion matrix of a classification model.
      * Public Health: Correlating socioeconomic status with specific health behaviors.
      * Marketing: Analyzing customer segment preferences across different product lines.
      * Social Science: Examining the relationship between education level and employment type.
      * Operations: Monitoring the frequency of specific defect types across manufacturing shifts.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the categorical variables to analyze.
    categorical_column_name_1 : str
        The name of the column representing the categorical variable for the y-axis (rows).
    categorical_column_name_2 : str
        The name of the column representing the categorical variable for the x-axis (columns).
    normalize_by : {"columns", "rows", "all"}, optional
        Defines how to calculate percentages in the contingency table. 
        "columns" calculates percentages within each column, "rows" within each row, 
        and "all" for the entire dataset. Defaults to "columns".
    color_palette : str, optional
        The seaborn/matplotlib color map to use for the heatmap (e.g., "Blues", 
        "viridis", "YlGnBu"). Defaults to "Blues".
    show_legend : bool, optional
        Whether to display the color bar legend on the right side of the plot. 
        Defaults to False.
    figure_size : tuple, optional
        The dimensions of the output figure as a (width, height) tuple in inches. 
        Defaults to (8, 6).
    data_label_format : str, optional
        The format string for the annotations inside the heatmap cells (e.g., ".1%" 
        for percentages or "d" for integers). Defaults to ".1%".
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
    # Epidemiology: Correlation between vaccine type and symptom severity
    import pandas as pd
    vax_df = pd.DataFrame({
        'Vaccine': ['Type A', 'Type B', 'Type A', 'Type B'] * 25,
        'Severity': ['Mild', 'Mild', 'Severe', 'Moderate'] * 25
    })
    PlotContingencyHeatmap(
        vax_df, 'Vaccine', 'Severity',
        normalize_by="rows",
        title_for_plot="Symptom Severity by Vaccine Type",
        subtitle_for_plot="Row-normalized distribution of self-reported side effects"
    )

    # Healthcare: Patient department vs. admission source
    hosp_df = pd.DataFrame({
        'Dept': ['Emergency', 'Surgery', 'Internal Med'] * 40,
        'Source': ['Referral', 'EMS', 'Walk-in', 'Clinic'] * 30
    })
    PlotContingencyHeatmap(
        hosp_df, 'Dept', 'Source',
        normalize_by="all",
        color_palette="rocket",
        title_for_plot="Patient Intake Composition",
        caption_for_plot="Based on Q4 hospital administrative data."
    )
    """
    # Ensure that the normalize_by argument is valid
    if normalize_by not in ["columns", "rows", "all"]:
        raise ValueError("The normalize_by argument must be 'columns', 'rows', or 'all'.")
    
    # Convert normalize_by to an acceptable pd.crosstab argument
    if normalize_by == "rows":
        normalize_by = "index"
    
    # Create contingency table
    contingency_table = pd.crosstab(
        dataframe[categorical_column_name_1],
        dataframe[categorical_column_name_2],
        normalize=normalize_by
    )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Plot contingency table in a heatmap using seaborn
    ax = sns.heatmap(
        contingency_table,
        cmap=color_palette,
        annot=True,
        fmt=data_label_format,
        linewidths=0.5,
        cbar=show_legend
    )
    
    # Wrap y axis label using textwrap
    wrapped_variable_name = "\n".join(textwrap.wrap(categorical_column_name_1, 30))  # String wrap the variable name
    ax.set_ylabel(wrapped_variable_name)
    
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
    wrapped_variable_name = "\n".join(textwrap.wrap(categorical_column_name_2, 30))  # String wrap the variable name
    ax.set_xlabel(wrapped_variable_name)
    
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

