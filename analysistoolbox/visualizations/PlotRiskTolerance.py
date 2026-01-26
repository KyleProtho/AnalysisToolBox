# Load packages
import pandas as pd
import os
from math import ceil
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def PlotRiskTolerance(simulated_values,
                      risk_tolerance,
                      # Risk tolerance arguments
                      lower_is_worse=True,
                      variable_name="Outcome",
                      observed_value=None,
                      risk_tolerance_label="Tolerance",
                      # Histogram formatting arguments
                      fill_color="#999999",
                      fill_transparency=0.6,
                      # Text formatting arguments
                      title_for_plot="Risk Tolerance",
                      subtitle_for_plot="Shows the simulated outcomes that are worse than the risk tolerance.",
                      caption_for_plot=None,
                      data_source_for_plot=None,
                      show_y_axis=False,
                      title_y_indent=1.1,
                      subtitle_y_indent=1.05,
                      caption_y_indent=-0.15,
                      # Plot formatting arguments
                      figure_size=(8, 6),
                      # Plot saving arguments
                      filepath_to_save_plot=None):
    """
    Generate a formatted histogram to visualize risk tolerance and simulated outcomes.

    This function creates a high-quality visualization of simulated data (e.g., from 
    a Monte Carlo simulation) against a specified risk tolerance threshold. It 
    highlights outcomes that exceed the tolerance levels, calculates the 
    probability of such occurrences, and optionally plots an observed real-world 
    value for comparison. The chart is designed for decision support in 
    uncertain environments.

    Risk tolerance plots are essential for:
      * Epidemiology: Estimating the probability of hospitalizations exceeding ICU capacity.
      * Healthcare: Assessing the risk of clinical wait times exceeding an operational threshold.
      * Intelligence Analysis: Visualizing the probability of enemy troop concentrations within target sectors.
      * Data Science: Evaluating the risk of model latency exceeding a service-level agreement (SLA).
      * Public Health: Estimating the probability of daily pollutant levels exceeding safety limits.
      * Finance: Assessing the Value at Risk (VaR) for an investment portfolio.
      * Project Management: Estimating the probability of a project exceeding its budget or timeline.
      * Quality Control: Assessing the risk of manufacturing defect rates exceeding quality standards.

    Parameters
    ----------
    simulated_values : array_like (numpy.ndarray or pd.Series)
        The collection of simulated data points to be visualized.
    risk_tolerance : int or float
        The threshold value representing the maximum acceptable risk level.
    lower_is_worse : bool, optional
        Whether values smaller than `risk_tolerance` are considered negative 
        outcomes. Defaults to True.
    variable_name : str, optional
        The label for the simulated metric (e.g., "Profit", "Cases"). 
        Defaults to "Outcome".
    observed_value : int or float, optional
        A specific actual or historical value to plot as a reference line. 
        Defaults to None.
    risk_tolerance_label : str, optional
        The text label used to identify the risk tolerance line. 
        Defaults to "Tolerance".
    fill_color : str, optional
        The hex color code for the non-critical portion of the histogram. 
        Defaults to "#999999".
    fill_transparency : float, optional
        The transparency level (alpha) of the histogram bars. Defaults to 0.6.
    title_for_plot : str, optional
        The primary title text at the top of the chart. Defaults to "Risk Tolerance".
    subtitle_for_plot : str, optional
        Descriptive subtitle text below the main title. 
        Defaults to "Shows the simulated outcomes that are worse than the risk tolerance.".
    caption_for_plot : str, optional
        Explanatory text or notes at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the data source, appended to the caption. Defaults to None.
    show_y_axis : bool, optional
        Whether to display the vertical frequency axis. Defaults to False.
    title_y_indent : float, optional
        Vertical offset for the title position. Defaults to 1.1.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position. Defaults to 1.05.
    caption_y_indent : float, optional
        Vertical offset for the caption position. Defaults to -0.15.
    figure_size : tuple, optional
        Dimensions of the output figure as a (width, height) tuple in inches. 
        Defaults to (8, 6).
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the risk tolerance plot using matplotlib and 
        optionally saves it to disk.

    Examples
    --------
    # Epidemiology: Risk of exceeding hospital bed capacity
    import numpy as np
    sim_cases = np.random.normal(450, 50, 1000)
    PlotRiskTolerance(
        sim_cases, risk_tolerance=550, lower_is_worse=False,
        variable_name="ICU Beds Required",
        title_for_plot="Hospital Capacity Risk Analysis",
        subtitle_for_plot="Simulated ICU demand vs. total available beds"
    )

    # Intelligence Analysis: Enemy troop concentration risk
    troop_sim = np.random.triangular(100, 250, 600, 1000)
    PlotRiskTolerance(
        troop_sim, risk_tolerance=400, lower_is_worse=False,
        variable_name="Troop Count",
        observed_value=320,
        title_for_plot="Tactical Threat Assessment",
        subtitle_for_plot="Probability of enemy strength exceeding sector defense capacity"
    )

    # Healthcare: Risk of patient wait times exceeding threshold
    wait_sim = np.random.exponential(15, 1000)
    PlotRiskTolerance(
        wait_sim, risk_tolerance=30, lower_is_worse=False,
        variable_name="Wait Time (min)",
        observed_value=12,
        title_for_plot="ER Operational Risk Monitoring",
        caption_for_plot="Risk defined as wait times exceeding 30-minute threshold."
    )
    """
    
    # Ensure that risk tolerance is numeric
    if not isinstance(risk_tolerance, (int, float)):
        raise ValueError("Risk tolerance must be numeric.")
    
    # Ensure that observed_value is numeric if it is provided
    if observed_value != None and not isinstance(observed_value, (int, float)):
        raise ValueError("Observed value must be numeric.")
    
    # If simulated_values is a pandas series or a list, convert it to a numpy array
    if isinstance(simulated_values, pd.Series) or isinstance(simulated_values, list):
        simulated_values = simulated_values.to_numpy()
    
    # Create dataframe with only the outcome variable and the simulated_values
    dataframe = pd.DataFrame(simulated_values, columns=[variable_name])
    
    # Create risk flag column name
    if lower_is_worse:
        risk_flag_col_name = "Is Below Risk Tolerance"
    else:
        risk_flag_col_name = "Is Above Risk Tolerance"
    
    # Add flag for whether the outcome is worse than the risk tolerance
    if lower_is_worse:
        dataframe[risk_flag_col_name] = dataframe[variable_name] < risk_tolerance
    else:
        dataframe[risk_flag_col_name] = dataframe[variable_name] > risk_tolerance
        
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create histogram using seaborn
    sns.histplot(
        data=dataframe,
        x=variable_name,
        color=fill_color,
        alpha=fill_transparency,
        hue=risk_flag_col_name,
        palette={True: "#FF0000", False: "#999999"}
    )
    
    # Remove top, left, and right spines. Set bottom spine to dark gray.
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#262626")
    
    # Remove the y-axis, and adjust the indent of the plot titles
    if show_y_axis == False:
        ax.axes.get_yaxis().set_visible(False)
        x_indent = 0.015
    else:
        x_indent = -0.005
    
    # Remove the y-axis label
    ax.set_ylabel(None)
    
    # Calculate the percentage of outcomes that are worse than the risk tolerance
    risk_probability = dataframe[risk_flag_col_name].mean()
    
    # Show the risk tolerance as a vertical line with a label and probability
    ax.axvline(
        x=risk_tolerance,
        ymax=0.97-.02,
        color="#262626",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5
    )
    ax.text(
        x=risk_tolerance, 
        y=plt.ylim()[1] * 0.97, 
        s=risk_tolerance_label+': {:.2f}'.format(risk_tolerance)+' ({:.1%})'.format(risk_probability),
        horizontalalignment='center',
        # fontname="Arial",
        fontsize=9,
        color="#262626",
        alpha=0.75
    )
    
    # Show the oberseved value as a vertical line with a label
    if observed_value != None:
        ax.axvline(
            x=observed_value,
            ymax=0.90-.02,
            color="#262626",
            linewidth=2,
            alpha=0.8
        )
        ax.text(
            x=observed_value, 
            y=plt.ylim()[1] * .90,
            s=variable_name+': {:.2f}'.format(observed_value),
            horizontalalignment='center',
            # fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
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
    
