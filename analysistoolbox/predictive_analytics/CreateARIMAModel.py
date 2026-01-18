# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import textwrap
import warnings

# Declare function
def CreateARIMAModel(dataframe,
                     outcome_column_name,
                     time_column_name=None,
                     # Model parameter arguments
                     lookback_periods=1,
                     differencing_periods=0,
                     lag_periods=1,
                     moving_average_periods=1,
                     # Output arguments
                     plot_time_series=False,
                     time_series_figure_size=(8, 5),
                     # Line formatting arguments
                     line_color="#3269a8",
                     line_alpha=0.8,
                     # Text formatting arguments
                     number_of_x_axis_ticks=None,
                     x_axis_tick_rotation=None,
                     title_for_plot="Time Series",
                     subtitle_for_plot="Shows the time series for the outcome variable.",
                     caption_for_plot=None,
                     data_source_for_plot=None,
                     x_indent=-0.127,
                     title_y_indent=1.125,
                     subtitle_y_indent=1.05,
                     caption_y_indent=-0.3,
                     # ARIMA plot arguments
                     test_for_stationarity=True,
                     show_acf_pacf_plots=False,
                     show_model_results=False,
                     plot_residuals=True):
    """
    Construct, fit, and evaluate an ARIMA (Autoregressive Integrated Moving Average) model.

    This function streamlines the time series analysis workflow by integrating 
    stationarity testing (Augmented Dickey-Fuller), visualization (line plots, 
    ACF/PACF), and model fitting using the statsmodels SARIMAX implementation. It 
    is designed to help analysts identify temporal patterns and build predictive 
    models for sequential data.

    Building ARIMA models is essential for:
      * Forecasting financial indicators like stock prices or currency exchange rates
      * Predicting supply chain requirements and inventory demand cycles
      * Analyzing intelligence trends such as frequency of security incidents over time
      * Monitoring economic throughput and identifying seasonal fluctuations
      * Projecting infrastructure load and server capacity requirements
      * Detecting anomalies and structural shifts in high-frequency sensor data
      * Evaluating the impact of historical interventions on future outcomes

    The function provides extensive plotting capabilities for both the raw input 
    series and model residuals (kernel density estimates). It also includes 
    integrated stationarity warnings to guide the selection of differencing 
    parameters (d).

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing the time series data to be modeled.
    outcome_column_name
        The name of the target variable column in the dataframe.
    time_column_name
        The name of the column representing the time axis (e.g., date, month). 
        Required if `plot_time_series` is True. Defaults to None.
    lookback_periods
        The number of lags to include in ACF and PACF plots. Defaults to 1.
    differencing_periods
        The 'd' parameter in ARIMA(p, d, q). The number of times the raw 
        observations are differenced. Defaults to 0.
    lag_periods
        The 'p' parameter in ARIMA(p, d, q). The number of lag observations 
        included in the model. Defaults to 1.
    moving_average_periods
        The 'q' parameter in ARIMA(p, d, q). The size of the moving average 
        window. Defaults to 1.
    plot_time_series
        Whether to generate a line plot of the outcome variable over time. 
        Defaults to False.
    time_series_figure_size
        Dimensions (width, height) of the time series plot. Defaults to (8, 5).
    line_color
        Hex color code for the time series line. Defaults to "#3269a8".
    line_alpha
        Transparency level for the plot line (0.0 to 1.0). Defaults to 0.8.
    number_of_x_axis_ticks
        Desired number of ticks on the x-axis. Defaults to None.
    x_axis_tick_rotation
        Degree of rotation for x-axis labels. Defaults to None.
    title_for_plot
        Main title for the generated visualization. Defaults to "Time Series".
    subtitle_for_plot
        Brief description shown below the main title. Defaults to 
        "Shows the time series for the outcome variable.".
    caption_for_plot
        Explanatory text displayed in the bottom margin. Defaults to None.
    data_source_for_plot
        Source citation displayed in the caption area. Defaults to None.
    x_indent, title_y_indent, subtitle_y_indent, caption_y_indent
        Coordinate offsets for precise text placement in the plot.
    test_for_stationarity
        If True, performs an Augmented Dickey-Fuller test and prints a 
        warning if the data appears non-stationary. Defaults to True.
    show_acf_pacf_plots
        Whether to display Autocorrelation and Partial Autocorrelation plots. 
        Defaults to False.
    show_model_results
        If True, prints a comprehensive summary of the fitted SARIMAX model. 
        Defaults to False.
    plot_residuals
        Whether to show a kernel density estimate (KDE) plot of the model 
        residuals. Defaults to True.

    Returns
    -------
    statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        The fitted ARIMA model object, containing coefficients, statistics, 
        and prediction methods.

    Examples
    --------
    # Create a simple ARIMA(1, 0, 1) model for sales data
    model = CreateARIMAModel(
        df, 
        outcome_column_name='Sales', 
        time_column_name='Date',
        plot_time_series=True
    )

    # Perform detailed analysis with differencing and ACF/PACF plots
    model = CreateARIMAModel(
        financial_df,
        outcome_column_name='Price',
        differencing_periods=1,
        show_acf_pacf_plots=True,
        show_model_results=True
    )

    """
    
    # If time series plot is requested, ensure time column is provided
    if plot_time_series:
        if time_column_name == None:
            raise Exception('A time column must be provided to plot the time series.')
    
    # Conduct ADF test to determine if data is stationary
    if test_for_stationarity:
        adfuller_test = adfuller(dataframe[outcome_column_name])
        adfuller_pvalue = adfuller_test[1]
        if adfuller_pvalue < 0.05:
            print('The data is stationary.')
        else:
            warnings.warn("The outcome variable is not stationary. Consider differencing the data.")
    
    # Plot time series, if requested
    if plot_time_series:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=time_series_figure_size)
        
        # Use Seaborn to create a line plot
        sns.lineplot(
            data=dataframe,
            x=time_column_name,
            y=outcome_column_name,
            color=line_color,
            alpha=line_alpha,
            ax=ax
        )
        
        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        # Set the number of ticks on the x-axis
        if number_of_x_axis_ticks is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(number_of_x_axis_ticks))
            
        # Rotate x-axis tick labels
        if x_axis_tick_rotation is not None:
            plt.xticks(rotation=x_axis_tick_rotation)

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
                fontsize=11,
                color="#666666",
                transform=ax.transAxes
            )
        
        # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
        ax.yaxis.set_label_coords(-0.1, 0.84)
        ax.yaxis.set_label_text(
            outcome_column_name,
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(
            time_column_name,
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
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )

        # Show plot
        plt.show()
    
    # Plot ACF and PACF plots
    if show_acf_pacf_plots:
        plot_acf(dataframe[outcome_column_name], lags=lookback_periods)
        plt.show()
        try:
            plot_pacf(dataframe[outcome_column_name], lags=lookback_periods)
            plt.show()
        except:
            print('Cannot product PACF plot for the lookback periods.')
            pass
        
    # Create SARIMAX model
    arima_model = SARIMAX(
        dataframe[outcome_column_name],
        order=(lag_periods, differencing_periods, moving_average_periods),
    )
    
    # Fit the model
    arima_model = arima_model.fit()
    
    # Show model results, if requested
    if show_model_results:
        print(arima_model.summary())
        
    # Plot residuals, if requested
    if plot_residuals:
        arima_model.resid.plot(kind='kde')
        plt.show()
    
    # Return the model
    return arima_model

