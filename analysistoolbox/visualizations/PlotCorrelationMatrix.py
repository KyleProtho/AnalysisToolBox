# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotCorrelationMatrix(dataframe,
                          list_of_value_column_names,
                          list_of_outcome_column_names=None,
                          show_as_pairplot=True,
                          # Pairplot formatting options
                          pairplot_size=(20, 20),
                          scatter_fill_color="#3269a8",
                          # Line formatting options
                          fit_lowess_line=True,
                          line_color="#cc4b5a",
                          # Text formatting arguments
                          title_for_plot="Pairplot of Numeric Variables",
                          subtitle_for_plot="Shows the linear relationship between numeric variables",
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          x_indent=-0.7,
                          title_y_indent=1.12,
                          title_font_size=14,
                          subtitle_y_indent=1.03,
                          subtitle_font_size=11,
                          caption_y_indent=-0.35,
                          caption_font_size=8,
                          # Plot saving arguments
                          filepath_to_save_plot=None):
    """
    Generate a formatted correlation matrix or multi-variable pairplot.

    This function produces either a numerical correlation matrix (printed to console) 
    or a high-quality visual pairplot using seaborn. When generating a pairplot, 
    it visualizes pairwise relationships between multiple numeric variables with 
    regression lines (optionally LOWESS), diagonal distributions, and thorough 
    formatting for professional reporting. It supports subsetting variables into 
    predictive "value" columns and "outcome" columns to focus the analysis.

    Correlation matrices and pairplots are essential for:
      * Epidemiology: Examining the relationships between multiple environmental factors and disease incidence.
      * Healthcare: Analyzing correlations between different patient vital signs and recovery outcomes.
      * Data Science: Identifying multi-collinearity during feature selection and exploratory data analysis.
      * Public Health: Correlating various socioeconomic indicators with regional life expectancy.
      * Finance: Analyzing the price movements of multiple assets to assess portfolio diversification.
      * Quality Control: Examining the relationship between different manufacturing sensor readings and defect counts.
      * Social Science: Exploring the connections between multiple demographic variables and survey scores.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the numeric data to analyze.
    list_of_value_column_names : list of str
        The names of the columns to be used on the x-axis of the pairplot (or the 
        entire variable set if `list_of_outcome_column_names` is None).
    list_of_outcome_column_names : list of str, optional
        The names of the outcome columns to be used on the y-axis of the pairplot. 
        If None, a symmetric grid is created using all value columns. Defaults to None.
    show_as_pairplot : bool, optional
        Whether to generate the visual pairplot. If False, the function prints 
        the numeric correlation matrix to the console instead. Defaults to True.
    pairplot_size : tuple, optional
        The dimensions of the entire figure grid in inches. Defaults to (20, 20).
    scatter_fill_color : str, optional
        The hex color code for the scatter plot points and diagonal distributions. 
        Defaults to "#3269a8".
    fit_lowess_line : bool, optional
        Whether to fit a Locally Weighted Scatterplot Smoothing (LOWESS) line to 
        the scatter plots to show non-linear trends. Defaults to True.
    line_color : str, optional
        The hex color code for the regression or LOWESS line. Defaults to "#cc4b5a".
    title_for_plot : str, optional
        The primary title text displayed at the top of the figure. 
        Defaults to "Pairplot of Numeric Variables".
    subtitle_for_plot : str, optional
        Descriptive subtitle text displayed below the title. 
        Defaults to "Shows the linear relationship between numeric variables".
    caption_for_plot : str, optional
        Explanatory text or notes displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot : str, optional
        Text identifying the source of the data, appended to the caption. Defaults to None.
    x_indent : float, optional
        Horizontal offset for titles and captions relative to the axes. Defaults to -0.7.
    title_y_indent : float, optional
        Vertical offset for the title position relative to the grid. Defaults to 1.12.
    title_font_size : int, optional
        Font size for the primary title. Defaults to 14.
    subtitle_y_indent : float, optional
        Vertical offset for the subtitle position relative to the grid. Defaults to 1.03.
    subtitle_font_size : int, optional
        Font size for the subtitle. Defaults to 11.
    caption_y_indent : float, optional
        Vertical offset for the caption position relative to the grid. Defaults to -0.35.
    caption_font_size : int, optional
        Font size for the caption text. Defaults to 8.
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the plot should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the plot using matplotlib (or prints a correlation 
        matrix) and optionally saves the visual output to disk.

    Examples
    --------
    # Epidemiology: Relationships between air quality indices
    import pandas as pd
    air_df = pd.DataFrame({
        'PM2.5': [12, 15, 45, 60, 10, 85],
        'PM10': [20, 25, 65, 80, 18, 110],
        'NO2': [5, 6, 22, 35, 4, 45],
        'Ozone': [30, 32, 28, 25, 31, 20]
    })
    PlotCorrelationMatrix(
        air_df, 
        list_of_value_column_names=['PM2.5', 'PM10', 'NO2', 'Ozone'],
        title_for_plot="Air Quality Indicator Correlations",
        pairplot_size=(8, 8)
    )

    # Healthcare: Vital sign correlations in emergency triage
    vitals_df = pd.DataFrame({
        'HR': [72, 85, 110, 60, 95],
        'BP_Sys': [120, 135, 160, 105, 140],
        'SpO2': [98, 97, 92, 99, 95],
        'Temp': [36.5, 37.2, 38.5, 36.8, 37.0]
    })
    PlotCorrelationMatrix(
        vitals_df, 
        list_of_value_column_names=['HR', 'BP_Sys', 'SpO2', 'Temp'],
        show_as_pairplot=False
    )
    # Prints the correlation matrix to the console instead of plotting
    """
    
    # Select relevant variables, keep complete cases only
    if list_of_outcome_column_names is None:
        completed_df = dataframe[list_of_value_column_names].dropna()
    else:
        completed_df = dataframe[list_of_outcome_column_names + list_of_value_column_names].dropna()

    # Drop Inf values
    completed_df = completed_df[np.isfinite(completed_df).all(1)]
    print("Count of complete observations for correlation matrix: " + str(len(completed_df.index)))
    
    # Show pairplot if specified -- otherwise, print correlation matrix
    if show_as_pairplot:
        plt.figure(figsize=pairplot_size)
        if list_of_outcome_column_names is None:
            ax = sns.pairplot(
                data=completed_df,
                kind='reg',
                markers='o',
                diag_kws={
                    'color': scatter_fill_color,
                    'alpha': 0.80
                },
                plot_kws={
                    'lowess': fit_lowess_line,
                    'line_kws': {
                        'color': line_color,
                        'alpha': 0.80
                    },
                    'scatter_kws': {
                        'color': scatter_fill_color,
                        'alpha': 0.30
                    }
                }
            )
            
        else:
            ax = sns.pairplot(
                data=completed_df,
                x_vars=list_of_value_column_names,
                y_vars=list_of_outcome_column_names,
                kind='reg',
                markers='o',
                diag_kws={
                    'color': scatter_fill_color,
                    'alpha': 0.80
                },
                plot_kws={
                    'lowess': fit_lowess_line,
                    'line_kws': {
                        'color': line_color,
                        'alpha': 0.80
                    },
                    'scatter_kws': {
                        'color': scatter_fill_color,
                        'alpha': 0.30
                    }
                }
            )
            
        # Wrap labels in each subplot
        for ax in ax.axes.flat:
            # x-axis labels
            x_labels = ax.get_xlabel()
            wrapped_x_labels = "\n".join(textwrap.wrap(x_labels, 40, break_long_words=False))
            ax.set_xlabel(wrapped_x_labels, horizontalalignment='center')
            # y-axis labels
            y_labels = ax.get_ylabel()
            wrapped_y_labels = "\n".join(textwrap.wrap(y_labels, 40, break_long_words=False))
            ax.set_ylabel(wrapped_y_labels, horizontalalignment='center')
        
        # Format tick labels to be Arial, size 9, and color #666666
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
        )
        
        # Adjust y-indents for title and subtitle
        if list_of_outcome_column_names is None:
            title_y_indent = title_y_indent*len(list_of_value_column_names)
            subtitle_y_indent = subtitle_y_indent*len(list_of_value_column_names)
        else:
            title_y_indent = title_y_indent*len(list_of_outcome_column_names)
            subtitle_y_indent = subtitle_y_indent*len(list_of_outcome_column_names)
        
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent*len(list_of_value_column_names),
            y=title_y_indent,
            s=title_for_plot,
            # fontname="Arial",
            fontsize=title_font_size,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent*len(list_of_value_column_names),
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            # fontname="Arial",
            fontsize=subtitle_font_size,
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
                if list_of_outcome_column_names is None:
                    wrapped_caption = textwrap.fill(caption_for_plot, 110, break_long_words=False)
                else:
                    wrapped_caption = textwrap.fill(caption_for_plot, 60, break_long_words=False)
                
            # Add the data source to the caption, if one is provided
            if data_source_for_plot != None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            
            # Add the caption to the plot
            ax.text(
                x=x_indent*len(list_of_value_column_names),
                y=caption_y_indent,
                s=wrapped_caption,
                # fontname="Arial",
                fontsize=caption_font_size,
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
        
        # Close plot
        plt.clf()
    
    else:
        print(completed_df.corr())

