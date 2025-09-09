# Minimal test snippet for CreateExponentialSmoothingModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the analysistoolbox module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysistoolbox'))

from analysistoolbox.predictive_analytics.CreateExponentialSmoothingModel import CreateExponentialSmoothingModel

# Create simple sample data
dates = pd.date_range(start='2020-01-01', periods=12, freq='M')
np.random.seed(42)
values = 100 + np.cumsum(np.random.randn(12)) + 5 * np.sin(np.arange(12) * 2 * np.pi / 12)

df = pd.DataFrame({'date': dates, 'value': values})

# Test the function
results = CreateExponentialSmoothingModel(
    dataframe=df,
    time_column='date',
    outcome_column='value',
    smoothing_type='auto',  # Let it choose the best model
    forecast_periods=3,
    print_model_performance=True,
    plot_model_performance=True,
    plot_forecast=True
)

print(f"Selected model: {results['model_type']}")
print(f"RMSE: {results['performance_metrics']['rmse']:.4f}")
