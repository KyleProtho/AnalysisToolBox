import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must precede any pyplot import

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pin to the local dev source tree (insert repo root so stdlib 'statistics' is not shadowed)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysistoolbox.predictive_analytics.CreateARIMAModel import CreateARIMAModel


def _make_df(seed=412, n=80):
    rng = np.random.default_rng(seed)
    # Stationary AR(1)-like process: y_t = 0.7 * y_{t-1} + noise
    y = np.zeros(n)
    noise = rng.standard_normal(n) * 0.5
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + noise[t]
    dates = pd.date_range('2020-01-01', periods=n, freq='MS')
    return pd.DataFrame({'date': dates, 'y': y})


class TestCreateARIMAModel(unittest.TestCase):

    def tearDown(self):
        plt.clf()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # Return type
    # ------------------------------------------------------------------ #

    def test_returns_fitted_model(self):
        """Returns a SARIMAXResultsWrapper object."""
        from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
        df = _make_df()
        result = CreateARIMAModel(
            df, 'y',
            test_for_stationarity=False,
            plot_residuals=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsInstance(result, SARIMAXResultsWrapper)

    def test_model_has_fittedvalues(self):
        """Fitted model exposes fittedvalues."""
        df = _make_df()
        result = CreateARIMAModel(
            df, 'y',
            test_for_stationarity=False,
            plot_residuals=False,
            plot_training_and_test_mse=False,
        )
        self.assertTrue(hasattr(result, 'fittedvalues'))

    # ------------------------------------------------------------------ #
    # Training and test RMSE output
    # ------------------------------------------------------------------ #

    def test_print_performance_includes_training_rmse(self):
        """print_model_training_performance=True prints 'Training RMSE:'."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                print_model_training_performance=True,
                plot_residuals=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training RMSE:', buf.getvalue())

    def test_print_performance_includes_test_rmse(self):
        """print_model_training_performance=True prints 'Test RMSE:'."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                print_model_training_performance=True,
                plot_residuals=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test RMSE:', buf.getvalue())

    def test_rmse_values_are_positive_finite(self):
        """Training RMSE and Test RMSE are positive, finite floats."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                print_model_training_performance=True,
                plot_residuals=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_rmse = float(next(l for l in lines if 'Training RMSE:' in l).split(':')[1])
        test_rmse  = float(next(l for l in lines if 'Test RMSE:' in l).split(':')[1])
        self.assertGreater(train_rmse, 0)
        self.assertGreater(test_rmse, 0)
        self.assertTrue(np.isfinite(train_rmse))
        self.assertTrue(np.isfinite(test_rmse))

    def test_print_performance_off_by_default(self):
        """With print_model_training_performance=False, no RMSE lines appear."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                print_model_training_performance=False,
                plot_residuals=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertNotIn('Training RMSE:', buf.getvalue())
        self.assertNotIn('Test RMSE:', buf.getvalue())

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_custom_test_size(self):
        """test_size=0.1 produces a valid model without error."""
        df = _make_df()
        result = CreateARIMAModel(
            df, 'y',
            test_size=0.1,
            test_for_stationarity=False,
            plot_residuals=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsNotNone(result)

    def test_differencing_periods(self):
        """differencing_periods=1 (ARIMA(1,1,1)) trains without error."""
        df = _make_df()
        result = CreateARIMAModel(
            df, 'y',
            differencing_periods=1,
            test_for_stationarity=False,
            plot_residuals=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsNotNone(result)

    # ------------------------------------------------------------------ #
    # Plot smoke tests (verify no exceptions are raised)
    # ------------------------------------------------------------------ #

    def test_rmse_comparison_plot_enabled(self):
        """plot_training_and_test_mse=True renders without raising an exception."""
        df = _make_df()
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                plot_residuals=False,
                plot_training_and_test_mse=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with RMSE comparison plot enabled: {e}")

    def test_rmse_comparison_plot_disabled(self):
        """plot_training_and_test_mse=False skips the chart without error."""
        df = _make_df()
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                plot_residuals=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with RMSE plot disabled: {e}")

    def test_custom_bar_colors(self):
        """Custom training_bar_color and test_bar_color are accepted without error."""
        df = _make_df()
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                plot_residuals=False,
                plot_training_and_test_mse=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")

    def test_residual_plot_smoke(self):
        """plot_residuals=True renders without raising an exception."""
        df = _make_df()
        try:
            CreateARIMAModel(
                df, 'y',
                test_for_stationarity=False,
                plot_residuals=True,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with residual plot enabled: {e}")

    def test_time_series_plot_smoke(self):
        """plot_time_series=True with a time column renders without raising an exception."""
        df = _make_df()
        try:
            CreateARIMAModel(
                df, 'y',
                time_column_name='date',
                plot_time_series=True,
                test_for_stationarity=False,
                plot_residuals=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with time series plot enabled: {e}")

    def test_all_plots_enabled(self):
        """Enabling all diagnostic plots raises no exception."""
        df = _make_df()
        try:
            CreateARIMAModel(
                df, 'y',
                time_column_name='date',
                plot_time_series=True,
                test_for_stationarity=False,
                plot_residuals=True,
                print_model_training_performance=True,
                plot_training_and_test_mse=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all plots enabled: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
