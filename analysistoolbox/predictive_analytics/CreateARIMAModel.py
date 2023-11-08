# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Declare function
def ConductARIMAModel(dataframe,
                      outcome_variable,
                      show_acf_pacf_plots=True,
                      lookback_periods=30,
                      differencing_periods=0,
                      lag_periods=0,
                      moving_average_periods=1,
                      show_model_results=True,
                      plot_residuals=True):
    # Conduct ADF test to determine if data is stationary
    adfuller_test = adfuller(dataframe[outcome_variable])
    adfuller_pvalue = adfuller_test[1]
    if adfuller_pvalue < 0.05:
        print('The data is stationary.')
    else:
        print('The data is not stationary.')
    
    # Plot ACF and PACF plots
    if show_acf_pacf_plots:
        plot_acf(dataframe[outcome_variable], lags=lookback_periods)
        plot_pacf(dataframe[outcome_variable], lags=lookback_periods)
        plt.show()
        
    # Create SARIMAX model
    arima_model = SARIMAX(
        dataframe[outcome_variable],
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

