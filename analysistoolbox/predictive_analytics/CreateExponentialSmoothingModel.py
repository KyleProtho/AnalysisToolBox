# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns  # Not used in this function
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import textwrap
import warnings
warnings.filterwarnings('ignore')

# Declare function
def CreateExponentialSmoothingModel(dataframe,
                                    time_column,
                                    outcome_column,
                                    # Model parameters
                                    smoothing_type="auto",  # "simple", "double", "triple", "auto"
                                    alpha=None,  # Smoothing parameter for level (0 < alpha <= 1)
                                    beta=None,   # Smoothing parameter for trend (0 < beta <= 1)
                                    gamma=None,  # Smoothing parameter for seasonality (0 < gamma <= 1)
                                    seasonal_periods=None,  # Number of periods in a season
                                    trend_type="additive",  # "additive" or "multiplicative"
                                    seasonal_type="additive",  # "additive" or "multiplicative"
                                    damped_trend=False,  # Whether to use damped trend
                                    # Model selection parameters
                                    auto_optimize=True,  # Whether to automatically optimize parameters
                                    optimization_method="L-BFGS-B",  # Optimization method
                                    # Output arguments
                                    print_model_performance=True,
                                    print_parameter_summary=True,
                                    print_forecast_summary=True,
                                    forecast_periods=12,  # Number of periods to forecast ahead
                                    # All plot arguments
                                    data_source_for_plot=None,
                                    # Model performance plot arguments
                                    plot_model_performance=True,
                                    plot_forecast=True,
                                    plot_decomposition=True,
                                    dot_fill_color="#999999",
                                    line_color="#b0170c",
                                    forecast_color="#ff6b35",
                                    figure_size_for_performance_plot=(12, 8),
                                    figure_size_for_forecast_plot=(12, 6),
                                    figure_size_for_decomposition_plot=(12, 8),
                                    title_for_performance_plot="Exponential Smoothing Model Performance",
                                    subtitle_for_performance_plot="Actual vs. Fitted values showing model accuracy",
                                    title_for_forecast_plot="Exponential Smoothing Forecast",
                                    subtitle_for_forecast_plot="Future predictions based on historical patterns",
                                    title_for_decomposition_plot="Time Series Decomposition",
                                    subtitle_for_decomposition_plot="Breakdown of trend, seasonal, and residual components",
                                    caption_for_performance_plot=None,
                                    caption_for_forecast_plot=None,
                                    caption_for_decomposition_plot=None,
                                    title_y_indent_for_performance_plot=1.10,
                                    subtitle_y_indent_for_performance_plot=1.05,
                                    title_y_indent_for_forecast_plot=1.10,
                                    subtitle_y_indent_for_forecast_plot=1.05,
                                    title_y_indent_for_decomposition_plot=1.10,
                                    subtitle_y_indent_for_decomposition_plot=1.05,
                                    caption_y_indent_for_performance_plot=-0.215,
                                    caption_y_indent_for_forecast_plot=-0.215,
                                    caption_y_indent_for_decomposition_plot=-0.215,
                                    x_indent_for_performance_plot=-0.115,
                                    x_indent_for_forecast_plot=-0.115,
                                    x_indent_for_decomposition_plot=-0.115):
    """
    Construct, fit, and evaluate exponential smoothing models for time series forecasting.

    This function provides a comprehensive interface for time series forecasting using 
    Exponential Smoothing techniques. It supports Simple (level), Double (level and trend), 
    and Triple (level, trend, and seasonality) exponential smoothing (Holt-Winters), 
    offering both manual parameter control and automated model selection and 
    optimization.

    Exponential smoothing is essential for:
      * Generating short-term demand forecasts for inventory management and logistics
      * Predicting future financial performance and revenue trends for budget planning
      * Analyzing seasonal patterns in economic indicators or market activity
      * Monitoring operational metrics and projecting future capacity requirements
      * Forecasting call volumes or service requests for workforce optimization
      * Smoothing noisy historical data to identify underlying structural trends
      * Evaluating the impact of seasonality on business performance metrics

    The function includes integrated tools for time series decomposition, performance 
    evaluation (RMSE, MAE, MAPE), and professional visualization of fitted values 
    and future forecasts. It automatically handles datetime indexing and offers 
    sophisticated plotting capabilities for model diagnostics.

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing the time series data.
    time_column
        The name of the column containing date or time information.
    outcome_column
        The name of the target variable column intended for forecasting.
    smoothing_type
        The complexity of the model: "simple" (level only), "double" (level + trend), 
        "triple" (level + trend + seasonality), or "auto" for heuristic-based 
        selection. Defaults to "auto".
    alpha
        The smoothing parameter for the level component (0 < alpha <= 1). 
        Adjusts the weight of recent observations. Defaults to None.
    beta
        The smoothing parameter for the trend component (0 < beta <= 1). 
        Determines how quickly the model adapts to trend changes. Defaults to None.
    gamma
        The smoothing parameter for the seasonal component (0 < gamma <= 1). 
        Controls the sensitivity to seasonal fluctuations. Defaults to None.
    seasonal_periods
        The number of observations that make up a full season (e.g., 12 for 
        monthly seasonality). Required for triple exponential smoothing.
    trend_type
        The nature of the trend component: "additive" or "multiplicative". 
        Defaults to "additive".
    seasonal_type
        The nature of the seasonal component: "additive" or "multiplicative". 
        Defaults to "additive".
    damped_trend
        Whether to dampen the trend approach to a horizontal line over time. 
        Defaults to False.
    auto_optimize
        If True, utilizes numerical solvers to find optimal smoothing 
        coefficients based on historical data. Defaults to True.
    optimization_method
        The solver used for parameter tuning (e.g., "L-BFGS-B"). 
        Defaults to "L-BFGS-B".
    print_model_performance, print_parameter_summary, print_forecast_summary
        Boolean flags to output statistical summaries to the console.
    forecast_periods
        The number of future time steps to project beyond the historical data. 
        Defaults to 12.
    data_source_for_plot
        Source citation string displayed in the caption of all generated plots. 
        Defaults to None.
    plot_model_performance, plot_forecast, plot_decomposition
        Boolean flags to control the generation of specific visualizations.
    dot_fill_color, line_color, forecast_color
        Aesthetic settings for the time series visualizations.
    figure_size_for_performance_plot, figure_size_for_forecast_plot, figure_size_for_decomposition_plot
        Dimensions (width, height) for the various output plots.
    title_for_performance_plot, subtitle_for_performance_plot, title_for_forecast_plot, subtitle_for_forecast_plot, title_for_decomposition_plot, subtitle_for_decomposition_plot
        Custom text elements for the plot titles and subtitles.
    caption_for_performance_plot, caption_for_forecast_plot, caption_for_decomposition_plot
        Explanatory text displayed in the plot margins.
    title_y_indent_for_performance_plot, subtitle_y_indent_for_performance_plot, title_y_indent_for_forecast_plot, subtitle_y_indent_for_forecast_plot, title_y_indent_for_decomposition_plot, subtitle_y_indent_for_decomposition_plot
        Vertical coordinate offsets for title and subtitle text.
    caption_y_indent_for_performance_plot, caption_y_indent_for_forecast_plot, caption_y_indent_for_decomposition_plot
        Vertical coordinate offsets for caption text.
    x_indent_for_performance_plot, x_indent_for_forecast_plot, x_indent_for_decomposition_plot
        Horizontal coordinate offsets for text placement in the plots.

    Returns
    -------
    dict
        A dictionary containing the following keys:
          * 'model': The fitted statsmodels Holt-Winters model object.
          * 'model_type': The smoothing method used (simple, double, or triple).
          * 'fitted_values': A Series of historical values predicted by the model.
          * 'forecast': A Series of future predictions.
          * 'performance_metrics': A dictionary with MSE, RMSE, MAE, and MAPE.
          * 'parameters': The optimized smoothing coefficients.
          * 'data': The cleaned time series data used for modeling.

    Examples
    --------
    # Generate a simple 12-period forecast with automated model selection
    results = CreateExponentialSmoothingModel(
        df, 
        time_column='Month', 
        outcome_column='Sales'
    )

    # Construct a custom Triple Exponential Smoothing model for quarterly data
    results = CreateExponentialSmoothingModel(
        financial_df,
        time_column='Quarter',
        outcome_column='Revenue',
        smoothing_type='triple',
        seasonal_periods=4,
        seasonal_type='multiplicative',
        forecast_periods=8,
        line_color='darkblue'
    )

    """
    
    # Create a copy of the dataframe to avoid modifying the original
    df = dataframe.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])
    
    # Sort by time column
    df = df.sort_values(time_column).reset_index(drop=True)
    
    # Set time column as index for time series analysis
    df_ts = df.set_index(time_column)
    
    # Extract the time series data
    ts_data = df_ts[outcome_column].dropna()
    
    # Check for sufficient data
    if len(ts_data) < 4:
        raise ValueError("Insufficient data for exponential smoothing. Need at least 4 observations.")
    
    # Auto-detect seasonal periods if not provided and using triple exponential smoothing
    if smoothing_type == "auto" or smoothing_type == "triple":
        if seasonal_periods is None:
            # Try to detect seasonal pattern
            if len(ts_data) >= 24:  # Need at least 2 years of monthly data
                # Check for common seasonal patterns
                for period in [12, 4, 52]:  # Monthly, quarterly, weekly
                    if len(ts_data) >= period * 2:
                        seasonal_periods = period
                        break
            else:
                seasonal_periods = min(4, len(ts_data) // 2)  # Default to 4 or half the data length
    
    # Auto-select smoothing type based on data characteristics
    if smoothing_type == "auto":
        # Perform seasonal decomposition to detect patterns
        try:
            if seasonal_periods and len(ts_data) >= seasonal_periods * 2:
                decomp = seasonal_decompose(ts_data, model='additive', period=seasonal_periods)
                seasonal_strength = np.var(decomp.seasonal) / np.var(ts_data)
                trend_strength = np.var(decomp.trend.dropna()) / np.var(ts_data)
                
                if seasonal_strength > 0.1:  # Strong seasonality
                    smoothing_type = "triple"
                elif trend_strength > 0.1:  # Strong trend
                    smoothing_type = "double"
                else:
                    smoothing_type = "simple"
            else:
                # Simple trend detection
                if len(ts_data) >= 10:
                    # Calculate simple trend
                    x = np.arange(len(ts_data))
                    slope = np.polyfit(x, ts_data.values, 1)[0]
                    if abs(slope) > np.std(ts_data) * 0.1:  # Significant trend
                        smoothing_type = "double"
                    else:
                        smoothing_type = "simple"
                else:
                    smoothing_type = "simple"
        except Exception:
            smoothing_type = "simple"  # Default fallback
    
    # Prepare model parameters
    model_params = {}
    
    if alpha is not None:
        model_params['smoothing_level'] = alpha
    if beta is not None:
        model_params['smoothing_trend'] = beta
    if gamma is not None:
        model_params['smoothing_seasonal'] = gamma
    
    # Set up the model based on smoothing type
    if smoothing_type == "simple":
        model = ExponentialSmoothing(
            ts_data,
            trend=None,
            seasonal=None,
            **model_params
        )
    elif smoothing_type == "double":
        model = ExponentialSmoothing(
            ts_data,
            trend=trend_type,
            seasonal=None,
            damped_trend=damped_trend,
            **model_params
        )
    elif smoothing_type == "triple":
        if seasonal_periods is None:
            raise ValueError("seasonal_periods must be specified for triple exponential smoothing")
        model = ExponentialSmoothing(
            ts_data,
            trend=trend_type,
            seasonal=seasonal_type,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            **model_params
        )
    else:
        raise ValueError("smoothing_type must be 'simple', 'double', 'triple', or 'auto'")
    
    # Fit the model
    if auto_optimize and not model_params:  # Only optimize if no parameters were manually set
        fitted_model = model.fit(optimized=True, method=optimization_method)
    else:
        fitted_model = model.fit()
    
    # Generate fitted values
    fitted_values = fitted_model.fittedvalues
    
    # Generate forecasts
    forecast = fitted_model.forecast(steps=forecast_periods)
    
    # Calculate performance metrics
    mse = mean_squared_error(ts_data, fitted_values)
    mae = mean_absolute_error(ts_data, fitted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((ts_data - fitted_values) / ts_data)) * 100
    
    # Print model performance if requested
    if print_model_performance:
        print(f"\n{'='*60}")
        print("EXPONENTIAL SMOOTHING MODEL PERFORMANCE")
        print(f"{'='*60}")
        print(f"Model Type: {smoothing_type.title()} Exponential Smoothing")
        print(f"Data Points: {len(ts_data)}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("="*60)
    
    # Print parameter summary if requested
    if print_parameter_summary:
        print(f"\n{'='*60}")
        print("MODEL PARAMETERS")
        print("="*60)
        params = fitted_model.params
        for param, value in params.items():
            # Handle different types of parameter values
            if isinstance(value, (int, float)):
                print(f"{param.replace('_', ' ').title()}: {value:.4f}")
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    print(f"{param.replace('_', ' ').title()}: {value.item():.4f}")
                else:
                    print(f"{param.replace('_', ' ').title()}: {value}")
            else:
                print(f"{param.replace('_', ' ').title()}: {value}")
        print("="*60)
    
    # Print forecast summary if requested
    if print_forecast_summary:
        print(f"\n{'='*60}")
        print("FORECAST SUMMARY")
        print("="*60)
        print(f"Forecast Periods: {forecast_periods}")
        print(f"Last Observed Value: {ts_data.iloc[-1]:.4f}")
        print(f"First Forecast Value: {forecast.iloc[0]:.4f}")
        print(f"Last Forecast Value: {forecast.iloc[-1]:.4f}")
        print(f"Forecast Range: {forecast.min():.4f} to {forecast.max():.4f}")
        print(f"{'='*60}")
    
    # Plot model performance if requested
    if plot_model_performance:
        plt.figure(figsize=figure_size_for_performance_plot)
        
        # Create the plot
        ax = plt.subplot(111)
        
        # Plot actual values
        ax.plot(ts_data.index, ts_data.values, 
                marker='o', markersize=3, linewidth=1, 
                color=dot_fill_color, alpha=0.7, label='Actual')
        
        # Plot fitted values
        ax.plot(fitted_values.index, fitted_values.values, 
                linewidth=2, color=line_color, label='Fitted')
        
        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        ax.tick_params(which='major', labelsize=9, color='#666666')
        ax.legend(loc='upper left', frameon=False, fontsize=10)
        
        # Set labels
        ax.set_ylabel(outcome_column, fontsize=10, color="#666666")
        ax.set_xlabel(time_column, fontsize=10, color="#666666")
        
        # Add title and subtitle
        ax.text(x=x_indent_for_performance_plot, y=title_y_indent_for_performance_plot,
                s=title_for_performance_plot, fontsize=14, color="#262626",
                transform=ax.transAxes)
        
        ax.text(x=x_indent_for_performance_plot, y=subtitle_y_indent_for_performance_plot,
                s=subtitle_for_performance_plot, fontsize=11, color="#666666",
                transform=ax.transAxes)
        
        # Add caption if provided
        if caption_for_performance_plot or data_source_for_plot:
            wrapped_caption = ""
            if caption_for_performance_plot:
                wrapped_caption = textwrap.fill(caption_for_performance_plot, 130, break_long_words=False)
            if data_source_for_plot:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            
            ax.text(x=x_indent_for_performance_plot, y=caption_y_indent_for_performance_plot,
                    s=wrapped_caption, fontsize=8, color="#666666",
                    transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()
    
    # Plot forecast if requested
    if plot_forecast:
        plt.figure(figsize=figure_size_for_forecast_plot)
        
        ax = plt.subplot(111)
        
        # Plot historical data
        ax.plot(ts_data.index, ts_data.values, 
                marker='o', markersize=3, linewidth=1, 
                color=dot_fill_color, alpha=0.7, label='Historical')
        
        # Plot fitted values
        ax.plot(fitted_values.index, fitted_values.values, 
                linewidth=2, color=line_color, label='Fitted')
        
        # Create forecast index
        last_date = ts_data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            if pd.infer_freq(ts_data.index) == 'D':  # Daily
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                             periods=forecast_periods, freq='D')
            elif pd.infer_freq(ts_data.index) == 'M':  # Monthly
                forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                             periods=forecast_periods, freq='M')
            elif pd.infer_freq(ts_data.index) == 'Q':  # Quarterly
                forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=3), 
                                             periods=forecast_periods, freq='Q')
            elif pd.infer_freq(ts_data.index) == 'Y':  # Yearly
                forecast_index = pd.date_range(start=last_date + pd.DateOffset(years=1), 
                                             periods=forecast_periods, freq='Y')
            else:
                # Default to monthly if frequency can't be inferred
                forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                             periods=forecast_periods, freq='M')
        else:
            # If not datetime, create simple numeric index
            forecast_index = range(len(ts_data), len(ts_data) + forecast_periods)
        
        # Plot forecast
        ax.plot(forecast_index, forecast.values, 
                linewidth=2, color=forecast_color, label='Forecast')
        
        # Add confidence interval if available
        try:
            conf_int = fitted_model.get_prediction(start=len(ts_data), 
                                                 end=len(ts_data) + forecast_periods - 1).conf_int()
            ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                           alpha=0.2, color=forecast_color, label='95% Confidence Interval')
        except Exception:
            pass  # Confidence intervals not available for all models
        
        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        ax.tick_params(which='major', labelsize=9, color='#666666')
        ax.legend(loc='upper left', frameon=False, fontsize=10)
        
        # Set labels
        ax.set_ylabel(outcome_column, fontsize=10, color="#666666")
        ax.set_xlabel(time_column, fontsize=10, color="#666666")
        
        # Add title and subtitle
        ax.text(x=x_indent_for_forecast_plot, y=title_y_indent_for_forecast_plot,
                s=title_for_forecast_plot, fontsize=14, color="#262626",
                transform=ax.transAxes)
        
        ax.text(x=x_indent_for_forecast_plot, y=subtitle_y_indent_for_forecast_plot,
                s=subtitle_for_forecast_plot, fontsize=11, color="#666666",
                transform=ax.transAxes)
        
        # Add caption if provided
        if caption_for_forecast_plot or data_source_for_plot:
            wrapped_caption = ""
            if caption_for_forecast_plot:
                wrapped_caption = textwrap.fill(caption_for_forecast_plot, 130, break_long_words=False)
            if data_source_for_plot:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            
            ax.text(x=x_indent_for_forecast_plot, y=caption_y_indent_for_forecast_plot,
                    s=wrapped_caption, fontsize=8, color="#666666",
                    transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()
    
    # Plot decomposition if requested and applicable
    if plot_decomposition and smoothing_type == "triple" and seasonal_periods:
        try:
            plt.figure(figsize=figure_size_for_decomposition_plot)
            
            # Perform seasonal decomposition
            decomp = seasonal_decompose(ts_data, model=seasonal_type, period=seasonal_periods)
            
            # Create subplots
            fig, axes = plt.subplots(4, 1, figsize=figure_size_for_decomposition_plot)
            fig.suptitle(title_for_decomposition_plot, fontsize=14, color="#262626", y=0.98)
            
            # Plot original data
            axes[0].plot(decomp.observed.index, decomp.observed.values, color=dot_fill_color, linewidth=1)
            axes[0].set_title('Original', fontsize=11, color="#666666")
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            
            # Plot trend
            axes[1].plot(decomp.trend.index, decomp.trend.values, color=line_color, linewidth=2)
            axes[1].set_title('Trend', fontsize=11, color="#666666")
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)
            
            # Plot seasonal
            axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color=forecast_color, linewidth=1)
            axes[2].set_title('Seasonal', fontsize=11, color="#666666")
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)
            
            # Plot residual
            axes[3].plot(decomp.resid.index, decomp.resid.values, color="#999999", linewidth=1)
            axes[3].set_title('Residual', fontsize=11, color="#666666")
            axes[3].spines['top'].set_visible(False)
            axes[3].spines['right'].set_visible(False)
            
            # Add subtitle
            fig.text(0.5, 0.92, subtitle_for_decomposition_plot, 
                    fontsize=11, color="#666666", ha='center')
            
            # Add caption if provided
            if caption_for_decomposition_plot or data_source_for_plot:
                wrapped_caption = ""
                if caption_for_decomposition_plot:
                    wrapped_caption = textwrap.fill(caption_for_decomposition_plot, 130, break_long_words=False)
                if data_source_for_plot:
                    wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
                
                fig.text(0.5, 0.02, wrapped_caption, fontsize=8, color="#666666", ha='center')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88, bottom=0.15)
            plt.show()
        except Exception as e:
            print(f"Could not create decomposition plot: {e}")
    
    # Return results
    results = {
        'model': fitted_model,
        'model_type': smoothing_type,
        'fitted_values': fitted_values,
        'forecast': forecast,
        'performance_metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        },
        'parameters': fitted_model.params,
        'data': ts_data
    }
    
    return results
