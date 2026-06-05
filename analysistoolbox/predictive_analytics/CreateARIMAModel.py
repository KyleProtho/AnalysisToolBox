# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
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
                     test_size=0.2,
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
                     plot_residuals=True,
                     # Performance output arguments
                     print_model_training_performance=False,
                     # RMSE comparison plot arguments
                     plot_training_and_test_mse=True,
                     training_bar_color="#3a86ff",
                     test_bar_color="#b0170c",
                     figure_size_for_mse_comparison_plot=(7, 5),
                     title_for_mse_comparison_plot=None,
                     subtitle_for_mse_comparison_plot=None,
                     caption_for_mse_comparison_plot=None,
                     title_y_indent_for_mse_comparison_plot=1.10,
                     subtitle_y_indent_for_mse_comparison_plot=1.05,
                     caption_y_indent_for_mse_comparison_plot=-0.15,
                     x_indent_for_mse_comparison_plot=-0.115):
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
    test_size
        Proportion of observations held out as a temporal test set (the last
        `test_size` fraction of rows). The model is fit on the remaining
        earlier observations. Defaults to 0.2.
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
        residuals (training set only). Defaults to True.
    print_model_training_performance
        If True, prints training RMSE and test RMSE after fitting.
        Defaults to False.
    plot_training_and_test_mse
        Whether to render a bar chart comparing training RMSE and test RMSE.
        Defaults to True.
    training_bar_color, test_bar_color
        Bar colors for the training and test bars. Defaults to blue / red.
    figure_size_for_mse_comparison_plot
        Dimensions (width, height) for the comparison chart. Defaults to (7, 5).
    title_for_mse_comparison_plot, subtitle_for_mse_comparison_plot, caption_for_mse_comparison_plot
        Text elements for the comparison chart. Sensible defaults are used when None.
    title_y_indent_for_mse_comparison_plot, subtitle_y_indent_for_mse_comparison_plot, caption_y_indent_for_mse_comparison_plot, x_indent_for_mse_comparison_plot
        Coordinate offsets for text placement in the comparison chart.

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

    # Sort chronologically if a time column is available
    if time_column_name is not None:
        dataframe = dataframe.sort_values(time_column_name).reset_index(drop=True)

    # Split data temporally: last test_size proportion becomes the test set
    n_total = len(dataframe)
    n_train = int(n_total * (1 - test_size))
    train_series = dataframe[outcome_column_name].iloc[:n_train]
    test_series = dataframe[outcome_column_name].iloc[n_train:]

    # Conduct ADF test to determine if data is stationary (uses full series for diagnostics)
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
        
    # Create SARIMAX model (fit on training portion only)
    arima_model = SARIMAX(
        train_series,
        order=(lag_periods, differencing_periods, moving_average_periods),
    )

    # Fit the model
    arima_model = arima_model.fit(disp=False)

    # Show model results, if requested
    if show_model_results:
        print(arima_model.summary())

    # Compute training and test RMSE
    training_residuals = arima_model.resid.dropna()
    training_rmse = np.sqrt((training_residuals ** 2).mean())
    forecast_result = arima_model.forecast(steps=len(test_series))
    test_rmse = np.sqrt(metrics.mean_squared_error(test_series.values, forecast_result.values))

    # Print training and test RMSE
    if print_model_training_performance:
        print('Training RMSE:', training_rmse)
        print('Test RMSE:', test_rmse)

    # Plot residuals, if requested
    if plot_residuals:
        arima_model.resid.plot(kind='kde')
        plt.show()

    # Plot training vs. test RMSE comparison
    if plot_training_and_test_mse:
        import textwrap as _tw
        chart_title = title_for_mse_comparison_plot or "Training vs. Test RMSE"
        chart_subtitle = subtitle_for_mse_comparison_plot or "Compares model error on the training and test datasets."

        fig, ax = plt.subplots(figsize=figure_size_for_mse_comparison_plot)
        bar_values = [training_rmse, test_rmse]
        bar_labels = ['Training', 'Test']
        bar_colors = [training_bar_color, test_bar_color]
        bars = ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.8, width=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                    '{:,.4f}'.format(height), ha='center', va='bottom', fontsize=10, color='#262626')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_visible(False)
        ax.tick_params(which='major', labelsize=9, color='#666666')
        ax.yaxis.set_ticks([])
        plt.subplots_adjust(top=0.85)
        ax.text(x=x_indent_for_mse_comparison_plot, y=title_y_indent_for_mse_comparison_plot,
                s=chart_title, fontsize=14, color="#262626", transform=ax.transAxes)
        ax.text(x=x_indent_for_mse_comparison_plot, y=subtitle_y_indent_for_mse_comparison_plot,
                s=chart_subtitle, fontsize=11, color="#666666", transform=ax.transAxes)
        if caption_for_mse_comparison_plot is not None:
            ax.text(x=x_indent_for_mse_comparison_plot, y=caption_y_indent_for_mse_comparison_plot,
                    s=_tw.fill(caption_for_mse_comparison_plot, 80, break_long_words=False),
                    fontsize=8, color="#666666", transform=ax.transAxes)
        plt.show()
        plt.clf()

    # Return the model
    return arima_model

