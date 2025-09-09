#!/usr/bin/env python3
"""
Simple test snippet for CreateExponentialSmoothingModel function
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the analysistoolbox module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysistoolbox'))

from analysistoolbox.predictive_analytics.CreateExponentialSmoothingModel import CreateExponentialSmoothingModel

# Create sample time series data
print("Creating sample time series data...")

# Create 24 months of monthly data with trend and seasonality
dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
np.random.seed(42)

# Generate data with trend and seasonality
trend = np.linspace(100, 150, 24)  # Upward trend
seasonal = 10 * np.sin(2 * np.pi * np.arange(24) / 12)  # Annual seasonality
noise = np.random.normal(0, 3, 24)  # Random noise
values = trend + seasonal + noise

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': values
})

print(f"Sample data shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())

# Test Simple Exponential Smoothing
print("\n" + "="*60)
print("TESTING SIMPLE EXPONENTIAL SMOOTHING")
print("="*60)

results_simple = CreateExponentialSmoothingModel(
    dataframe=df,
    time_column='date',
    outcome_column='sales',
    smoothing_type='simple',
    forecast_periods=6,
    print_model_performance=True,
    print_parameter_summary=True,
    print_forecast_summary=True,
    plot_model_performance=True,
    plot_forecast=True,
    plot_decomposition=False  # Skip decomposition for simple model
)

print(f"\nModel type used: {results_simple['model_type']}")
print(f"RMSE: {results_simple['performance_metrics']['rmse']:.4f}")
print(f"MAPE: {results_simple['performance_metrics']['mape']:.2f}%")

# Test Double Exponential Smoothing
print("\n" + "="*60)
print("TESTING DOUBLE EXPONENTIAL SMOOTHING")
print("="*60)

results_double = CreateExponentialSmoothingModel(
    dataframe=df,
    time_column='date',
    outcome_column='sales',
    smoothing_type='double',
    forecast_periods=6,
    print_model_performance=True,
    print_parameter_summary=True,
    print_forecast_summary=True,
    plot_model_performance=True,
    plot_forecast=True,
    plot_decomposition=False
)

print(f"\nModel type used: {results_double['model_type']}")
print(f"RMSE: {results_double['performance_metrics']['rmse']:.4f}")
print(f"MAPE: {results_double['performance_metrics']['mape']:.2f}%")

# Test Triple Exponential Smoothing
print("\n" + "="*60)
print("TESTING TRIPLE EXPONENTIAL SMOOTHING")
print("="*60)

results_triple = CreateExponentialSmoothingModel(
    dataframe=df,
    time_column='date',
    outcome_column='sales',
    smoothing_type='triple',
    seasonal_periods=12,  # Monthly data with annual seasonality
    forecast_periods=6,
    print_model_performance=True,
    print_parameter_summary=True,
    print_forecast_summary=True,
    plot_model_performance=True,
    plot_forecast=True,
    plot_decomposition=True  # Show decomposition for triple model
)

print(f"\nModel type used: {results_triple['model_type']}")
print(f"RMSE: {results_triple['performance_metrics']['rmse']:.4f}")
print(f"MAPE: {results_triple['performance_metrics']['mape']:.2f}%")

# Test Auto Model Selection
print("\n" + "="*60)
print("TESTING AUTO MODEL SELECTION")
print("="*60)

results_auto = CreateExponentialSmoothingModel(
    dataframe=df,
    time_column='date',
    outcome_column='sales',
    smoothing_type='auto',  # Let the function choose the best model
    forecast_periods=6,
    print_model_performance=True,
    print_parameter_summary=True,
    print_forecast_summary=True,
    plot_model_performance=True,
    plot_forecast=True,
    plot_decomposition=True
)

print(f"\nAuto-selected model type: {results_auto['model_type']}")
print(f"RMSE: {results_auto['performance_metrics']['rmse']:.4f}")
print(f"MAPE: {results_auto['performance_metrics']['mape']:.2f}%")

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*60)
