"""
Test script for CreateSLURPDistributionFromExponentialSmoothing function

Note: There may be a type check issue in the function (line 117) that checks for
ExponentialSmoothing class instead of the fitted result. If the test fails with
a type error, the type check in the function may need to be fixed.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from analysistoolbox.simulations.CreateSLURPDistributionFromExponentialSmoothing import CreateSLURPDistributionFromExponentialSmoothing

# Set random seed for reproducibility
np.random.seed(42)

# Create sample time series data
# Generate a time series with trend and seasonality
n_periods = 50
time_index = pd.date_range(start='2020-01-01', periods=n_periods, freq='M')
trend = np.linspace(100, 150, n_periods)
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
noise = np.random.normal(0, 5, n_periods)
data = trend + seasonal + noise

# Create a pandas Series
ts_data = pd.Series(data, index=time_index)

# Fit exponential smoothing model
# Using additive trend and seasonal components
model = ExponentialSmoothing(
    ts_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12
)
fitted_model = model.fit()

print(f"Fitted model type: {type(fitted_model)}")
print(f"Model has forecast method: {hasattr(fitted_model, 'forecast')}")
print(f"Model has resid attribute: {hasattr(fitted_model, 'resid')}")
print()

# Test 1: Basic usage with default parameters
print("Test 1: Basic usage with default parameters")
print("-" * 50)
result1 = CreateSLURPDistributionFromExponentialSmoothing(
    exponential_smoothing_model=fitted_model,
    show_distribution_plot=True,
    title_for_plot="SLURP Distribution from Exponential Smoothing",
    subtitle_for_plot="Basic test with default parameters"
)
print(f"Result shape: {result1.shape}")
print(f"Result statistics:")
print(result1.describe())
print("\n")

# Test 2: Custom parameters
print("Test 2: Custom parameters")
print("-" * 50)
result2 = CreateSLURPDistributionFromExponentialSmoothing(
    exponential_smoothing_model=fitted_model,
    forecast_steps=3,
    number_of_trials=5000,
    prediction_interval=0.90,
    show_distribution_plot=True,
    show_summary=True,
    title_for_plot="SLURP Distribution - Custom Parameters",
    subtitle_for_plot="3-step ahead forecast, 90% prediction interval"
)
print(f"Result shape: {result2.shape}")
print(f"Mean: {result2['forecast_value'].mean():.2f}")
print(f"Median: {result2['forecast_value'].median():.2f}")
print(f"Std: {result2['forecast_value'].std():.2f}")
print("\n")

# Test 3: With bounds
print("Test 3: With lower and upper bounds")
print("-" * 50)
result3 = CreateSLURPDistributionFromExponentialSmoothing(
    exponential_smoothing_model=fitted_model,
    lower_bound=0,
    upper_bound=200,
    show_distribution_plot=True,
    title_for_plot="SLURP Distribution with Bounds",
    subtitle_for_plot="Bounded between 0 and 200"
)
print(f"Result shape: {result3.shape}")
print(f"Min: {result3['forecast_value'].min():.2f}")
print(f"Max: {result3['forecast_value'].max():.2f}")
print("\n")

# Test 4: Return as array
print("Test 4: Return as array format")
print("-" * 50)
result4 = CreateSLURPDistributionFromExponentialSmoothing(
    exponential_smoothing_model=fitted_model,
    return_format='array',
    show_distribution_plot=False
)
print(f"Result type: {type(result4)}")
print(f"Result shape: {result4.shape}")
print(f"First 10 values: {result4[:10]}")
print("\n")

# Test 5: No plot
print("Test 5: No plot, just return data")
print("-" * 50)
result5 = CreateSLURPDistributionFromExponentialSmoothing(
    exponential_smoothing_model=fitted_model,
    show_distribution_plot=False,
    number_of_trials=1000
)
print(f"Result shape: {result5.shape}")
print(f"Summary statistics:")
print(result5.describe())

print("\n" + "=" * 50)
print("All tests completed!")

