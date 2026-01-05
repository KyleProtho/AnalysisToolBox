# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

# Declare function
def AddTukeyOutlierColumn(dataframe, 
                          value_column_name,
                          tukey_boundary_multiplier=1.5,
                          plot_tukey_outliers=False,
                          # Histogram formatting arguments
                          fill_color="#999999",
                          fill_transparency=0.6,
                          # Text formatting arguments
                          title_for_plot="Tukey Outliers",
                          subtitle_for_plot="Shows the values that are Tukey outliers.",
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          show_y_axis=False,
                          title_y_indent=1.1,
                          subtitle_y_indent=1.05,
                          caption_y_indent=-0.15,
                          # Plot formatting arguments
                          figure_size=(8, 6)):
    """
    Identify and flag statistical outliers using Tukey's method with optional visualization.

    This function applies Tukey's fence method (also known as the IQR method) to detect outliers
    in a numeric column. It calculates the interquartile range (IQR) and flags values that fall
    outside the boundaries defined by Q1 - k*IQR and Q3 + k*IQR, where k is the boundary
    multiplier (typically 1.5 for outliers or 3.0 for extreme outliers). A new binary column
    is added to indicate whether each value is an outlier (1) or not (0).

    The function is particularly useful for:
      * Data quality assessment and anomaly detection
      * Identifying extreme values in financial and scientific data
      * Preprocessing data for statistical modeling
      * Finding fraudulent transactions or unusual patterns
      * Quality control in manufacturing and operations
      * Exploratory data analysis and distribution understanding
      * Detecting measurement errors or data entry mistakes

    Tukey's method is robust and non-parametric, making it suitable for data that may not
    follow a normal distribution. The optional visualization feature creates a histogram
    with color-coded outliers and clearly marked boundary lines, making it easy to visually
    assess the distribution and outlier patterns.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the data to analyze for outliers. A new column will
        be added to flag outlier values.
    value_column_name
        Name of the numeric column to check for outliers. This column must contain numeric
        values (int or float).
    tukey_boundary_multiplier
        The multiplier for the IQR to determine outlier boundaries. Common values are 1.5
        (standard outliers) or 3.0 (extreme outliers). Lower values detect more outliers;
        higher values are more conservative. Defaults to 1.5.
    plot_tukey_outliers
        If True, generates a histogram visualization showing the distribution with outliers
        highlighted in red and boundary lines marked. If False, no plot is created.
        Defaults to False.
    fill_color
        Hex color code for non-outlier histogram bars when plotting. Only used if
        plot_tukey_outliers=True. Defaults to '#999999' (gray).
    fill_transparency
        Alpha transparency value for histogram bars (0.0 to 1.0). Lower values are more
        transparent. Only used if plot_tukey_outliers=True. Defaults to 0.6.
    title_for_plot
        Main title text for the plot. Only used if plot_tukey_outliers=True.
        Defaults to 'Tukey Outliers'.
    subtitle_for_plot
        Subtitle text displayed below the title. Only used if plot_tukey_outliers=True.
        Defaults to 'Shows the values that are Tukey outliers.'.
    caption_for_plot
        Optional caption text displayed at the bottom of the plot. Use for additional
        context or notes. Only used if plot_tukey_outliers=True. Defaults to None.
    data_source_for_plot
        Optional data source attribution text appended to the caption. Only used if
        plot_tukey_outliers=True. Defaults to None.
    show_y_axis
        If True, displays the y-axis (frequency counts) on the plot. If False, hides
        the y-axis for a cleaner appearance. Only used if plot_tukey_outliers=True.
        Defaults to False.
    title_y_indent
        Vertical position adjustment for the title as a fraction of the plot height.
        Only used if plot_tukey_outliers=True. Defaults to 1.1.
    subtitle_y_indent
        Vertical position adjustment for the subtitle as a fraction of the plot height.
        Only used if plot_tukey_outliers=True. Defaults to 1.05.
    caption_y_indent
        Vertical position adjustment for the caption as a fraction of the plot height.
        Only used if plot_tukey_outliers=True. Defaults to -0.15.
    figure_size
        Tuple specifying the (width, height) of the plot in inches. Only used if
        plot_tukey_outliers=True. Defaults to (8, 6).

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an additional binary column named
        '{value_column_name} - Tukey outlier' where 1 indicates an outlier and 0
        indicates a normal value. If plot_tukey_outliers=True, a visualization is
        also displayed.

    Examples
    --------
    # Detect outliers in transaction amounts with default settings
    import pandas as pd
    transactions = pd.DataFrame({
        'transaction_id': range(1, 11),
        'amount': [100, 150, 120, 130, 140, 110, 125, 1500, 135, 145]
    })
    transactions = AddTukeyOutlierColumn(transactions, 'amount')
    # Adds column 'amount - Tukey outlier' with 1 for the $1500 transaction

    # Visualize outliers with a plot
    sales_data = pd.DataFrame({
        'sale_price': [250, 275, 300, 280, 290, 310, 5000, 260, 270, 295]
    })
    sales_data = AddTukeyOutlierColumn(
        sales_data,
        'sale_price',
        plot_tukey_outliers=True,
        title_for_plot='Sales Price Outlier Detection',
        subtitle_for_plot='Identifying unusual sale prices'
    )
    # Creates plot showing distribution with outlier boundaries marked

    # Use stricter threshold for extreme outliers only
    quality_scores = pd.DataFrame({
        'quality_metric': [85, 88, 90, 87, 89, 91, 86, 45, 88, 92]
    })
    quality_scores = AddTukeyOutlierColumn(
        quality_scores,
        'quality_metric',
        tukey_boundary_multiplier=3.0  # More conservative
    )
    # Only flags very extreme values as outliers

    # Customize plot appearance with colors and captions
    response_times = pd.DataFrame({
        'response_ms': [120, 135, 142, 128, 138, 2500, 131, 125, 140, 129]
    })
    response_times = AddTukeyOutlierColumn(
        response_times,
        'response_ms',
        plot_tukey_outliers=True,
        fill_color='#3498db',
        fill_transparency=0.7,
        show_y_axis=True,
        title_for_plot='API Response Time Analysis',
        subtitle_for_plot='Detecting performance anomalies',
        caption_for_plot='Red bars indicate outlier response times that may require investigation.',
        data_source_for_plot='Production API logs - January 2023',
        figure_size=(10, 7)
    )

    """
    
    # Calculate the IQR
    Q1 = dataframe[value_column_name].quantile(0.25)
    Q3 = dataframe[value_column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define the outlier bounds
    lower_bound = Q1 - tukey_boundary_multiplier * IQR
    upper_bound = Q3 + tukey_boundary_multiplier * IQR

    # Flag outliers
    dataframe[f'{value_column_name} - Tukey outlier'] = ((dataframe[value_column_name] < lower_bound) | (dataframe[value_column_name] > upper_bound)).astype(int)
    
    # Plot outliers, if requested
    if plot_tukey_outliers:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create histogram using seaborn
        sns.histplot(
            data=dataframe,
            x=value_column_name,
            color=fill_color,
            alpha=fill_transparency,
            hue=f'{value_column_name} - Tukey outlier',
            palette={True: "#FF0000", False: "#999999"}
        )
        
        # Remove the legend
        ax.get_legend().remove()
        
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
        
        # Show the lower boundary as a vertical line with a label
        ax.axvline(
            x=lower_bound,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=lower_bound, 
            y=plt.ylim()[1] * 0.97, 
            s='Lower bound: {:.2f}'.format(lower_bound),
            horizontalalignment='center',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
        )
        
        # Show the upper boundary as a vertical line with a label
        ax.axvline(
            x=upper_bound,
            ymax=0.97-.02,
            color="#262626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5
        )
        ax.text(
            x=upper_bound, 
            y=plt.ylim()[1] * 0.97, 
            s='Upper bound: {:.2f}'.format(upper_bound),
            horizontalalignment='center',
            fontname="Arial",
            fontsize=9,
            color="#262626",
            alpha=0.75
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
        plt.xticks(fontname='Arial')
        
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
        
        # Clear plot
        plt.clf()
    
    # Return the updated dataframe
    return dataframe

