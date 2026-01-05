# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotFunction(f_of_x, 
                 minimum_x : int =-10, 
                 maximum_x : int =10, 
                 n : int =100,
                 # Line formatting arguments
                 line_color : str ="#3269a8",
                 line_alpha : float =0.8,
                 markers : str ="o",
                 # Text formatting arguments
                 x_axis_variable_name : str ="x",
                 y_axis_variable_name : str ="f(x)",
                 title_for_plot : str ="Function Plot",
                 subtitle_for_plot : str ="Shows the function of x",
                 caption_for_plot : str =None,
                 data_source_for_plot : str =None,
                 x_indent : float =-0.127,
                 title_y_indent : float =1.125,
                 subtitle_y_indent : float =1.05,
                 caption_y_indent : float =-0.3,
                 # Plot formatting arguments
                 figure_size : tuple = (8, 5)):
    """
    Plot a real-valued function of a single variable over a numeric range.

    PlotFunction takes a callable (e.g., a lambda or function) that maps
    numeric inputs to numeric outputs and produces a line plot of the
    relationship between input and output over the interval [x_min, x_max].

    Plotting functions helps analysts visualize *behavioral patterns* of
    models, expose non-linear features such as peaks, troughs, plateaus,
    and inflection regions, and communicate insights. In practical analytic
    workflows — whether in intelligence analysis, criminal investigations,
    or healthcare analytics — visualizing how a model or measurement evolves
    with its input can be more informative than reporting raw numbers alone.

    Parameters
    ----------
        f_of_x
            The function of x to plot.
        minimum_x
            The minimum value of x to use when plotting the function. Defaults to -10.
        maximum_x
            The maximum value of x to use when plotting the function. Defaults to 10.
        n
            The number of points to use when plotting the function. Defaults to 100.
        line_color
            The color of the function plot line. Defaults to "#3269a8".
        line_alpha
            The alpha (transparency) of the function plot line. Defaults to 0.8.
        markers
            The marker style to use for the function plot. Defaults to "o".
        x_axis_variable_name
            The label for the x-axis. Defaults to "x".
        y_axis_variable_name
            The label for the y-axis. Defaults to "f(x)".
        title_for_plot
            The title for the plot. Defaults to "Function Plot".
        subtitle_for_plot
            The subtitle for the plot. Defaults to "Shows the function of x".
        caption_for_plot
            The caption for the plot. Defaults to None.
        data_source_for_plot
            The data source for the plot. Defaults to None.
        x_indent
            The x-indent for the x-axis label. Defaults to -0.127.
        title_y_indent
            The y-indent for the title. Defaults to 1.125.
        subtitle_y_indent
            The y-indent for the subtitle. Defaults to 1.05.
        caption_y_indent
            The y-indent for the caption. Defaults to -0.3.
        figure_size
            The size of the plot figure. Defaults to (8, 5).

    Returns
    -------
    None — the function displays a plot.

    Examples
    --------
    PlotFunction(
        f_of_x=lambda x: x**2,
        minimum_x=-10,
        maximum_x=10,
        n=100,
        line_color="#3269a8",
        line_alpha=0.8,
        markers="o",
        x_axis_variable_name="x",
        y_axis_variable_name="f(x)",
        title_for_plot="Function Plot",
        subtitle_for_plot="Shows the function of x",
        caption_for_plot=None,
        data_source_for_plot=None,
        x_indent=-0.127,
        title_y_indent=1.125,
        subtitle_y_indent=1.05,
        caption_y_indent=-0.3,
        figure_size=(8, 5)
    )

    Teaching Note
    -------------
    Plotting a function reveals *qualitative features* — curvature, growth
    rate changes, local maxima/minima, and boundaries — in ways that raw
    numerical output cannot. Visual inspection remains a cornerstone of
    analytic sense-making across disciplines.
    """
    
    # Plot the function
    x = np.linspace(minimum_x, maximum_x, n)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = f_of_x(x[i])
    
    # Convert the function to a dataframe
    dataframe = pd.DataFrame({
        "x": x,
        "y": y
    })
        
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create line plot with seaborn
    ax = sns.lineplot(
        data=dataframe,
        x=x,
        y=y,
        color=line_color,
        alpha=line_alpha,
        marker=markers,
        ax=ax
    )
    
    # Remove top and right spines, and set bottom and left spines to gray
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')

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
        fontname="Arial",
        fontsize=14,
        color="#262626",
        transform=ax.transAxes
    )
    
    # Word wrap the subtitle without splitting words
    if subtitle_for_plot != None:   
        subtitle_for_plot = textwrap.fill(subtitle_for_plot, 100, break_long_words=False)
    # Set the subtitle with Arial font, size 11, and color #666666
    ax.text(
        x=x_indent,
        y=subtitle_y_indent,
        s=subtitle_for_plot,
        fontname="Arial",
        fontsize=11,
        color="#666666",
        transform=ax.transAxes
    )
    
    # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
    ax.yaxis.set_label_coords(-0.1, 0.84)
    ax.yaxis.set_label_text(
        y_axis_variable_name,
        fontname="Arial",
        fontsize=10,
        color="#666666"
    )
    
    # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
    ax.xaxis.set_label_coords(0.9, -0.1)
    ax.xaxis.set_label_text(
        x_axis_variable_name,
        fontname="Arial",
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
            fontname="Arial",
            fontsize=8,
            color="#666666",
            transform=ax.transAxes
        )

    # Show plot
    plt.show()

