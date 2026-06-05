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
from analysistoolbox.predictive_analytics.CreateExponentialSmoothingModel import CreateExponentialSmoothingModel


def _make_df(seed=412, n=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2018-01-01', periods=n, freq='MS')
    # Simple trend + noise, no seasonality
    y = np.arange(n) * 0.5 + rng.standard_normal(n) * 2.0 + 10.0
    return pd.DataFrame({'date': dates, 'y': y})


class TestCreateExponentialSmoothingModel(unittest.TestCase):

    def tearDown(self):
        plt.clf()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # Return type
    # ------------------------------------------------------------------ #

    def test_returns_dict(self):
        """Function returns a dict with the expected keys."""
        df = _make_df()
        result = CreateExponentialSmoothingModel(
            df, 'date', 'y',
            smoothing_type='simple',
            print_model_performance=False,
            print_parameter_summary=False,
            print_forecast_summary=False,
            plot_model_performance=False,
            plot_forecast=False,
            plot_decomposition=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsInstance(result, dict)
        for key in ('model', 'model_type', 'fitted_values', 'forecast', 'performance_metrics', 'parameters', 'data'):
            self.assertIn(key, result)

    def test_performance_metrics_keys(self):
        """performance_metrics dict contains training_rmse and test_rmse."""
        df = _make_df()
        result = CreateExponentialSmoothingModel(
            df, 'date', 'y',
            smoothing_type='simple',
            print_model_performance=False,
            print_parameter_summary=False,
            print_forecast_summary=False,
            plot_model_performance=False,
            plot_forecast=False,
            plot_decomposition=False,
            plot_training_and_test_mse=False,
        )
        pm = result['performance_metrics']
        self.assertIn('training_rmse', pm)
        self.assertIn('test_rmse', pm)

    # ------------------------------------------------------------------ #
    # Training and test RMSE output
    # ------------------------------------------------------------------ #

    def test_print_performance_includes_training_rmse(self):
        """print_model_performance=True prints 'Training RMSE:'."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=True,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=False,
                plot_forecast=False,
                plot_decomposition=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training RMSE:', buf.getvalue())

    def test_print_performance_includes_test_rmse(self):
        """print_model_performance=True prints 'Test RMSE:'."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=True,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=False,
                plot_forecast=False,
                plot_decomposition=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test RMSE:', buf.getvalue())

    def test_rmse_values_are_positive_finite(self):
        """training_rmse and test_rmse in performance_metrics are positive finite floats."""
        df = _make_df()
        result = CreateExponentialSmoothingModel(
            df, 'date', 'y',
            smoothing_type='simple',
            print_model_performance=False,
            print_parameter_summary=False,
            print_forecast_summary=False,
            plot_model_performance=False,
            plot_forecast=False,
            plot_decomposition=False,
            plot_training_and_test_mse=False,
        )
        pm = result['performance_metrics']
        self.assertGreater(pm['training_rmse'], 0)
        self.assertGreater(pm['test_rmse'], 0)
        self.assertTrue(np.isfinite(pm['training_rmse']))
        self.assertTrue(np.isfinite(pm['test_rmse']))

    def test_print_performance_off_suppresses_rmse(self):
        """With print_model_performance=False, no Training/Test RMSE lines appear."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=False,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=False,
                plot_forecast=False,
                plot_decomposition=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertNotIn('Training RMSE:', buf.getvalue())
        self.assertNotIn('Test RMSE:', buf.getvalue())

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_double_smoothing(self):
        """smoothing_type='double' trains without error and returns RMSE metrics."""
        df = _make_df()
        result = CreateExponentialSmoothingModel(
            df, 'date', 'y',
            smoothing_type='double',
            print_model_performance=False,
            print_parameter_summary=False,
            print_forecast_summary=False,
            plot_model_performance=False,
            plot_forecast=False,
            plot_decomposition=False,
            plot_training_and_test_mse=False,
        )
        self.assertIn('training_rmse', result['performance_metrics'])

    def test_custom_test_size(self):
        """test_size=0.1 produces valid RMSE metrics without error."""
        df = _make_df()
        result = CreateExponentialSmoothingModel(
            df, 'date', 'y',
            smoothing_type='simple',
            test_size=0.1,
            print_model_performance=False,
            print_parameter_summary=False,
            print_forecast_summary=False,
            plot_model_performance=False,
            plot_forecast=False,
            plot_decomposition=False,
            plot_training_and_test_mse=False,
        )
        self.assertGreater(result['performance_metrics']['test_rmse'], 0)

    # ------------------------------------------------------------------ #
    # Plot smoke tests (verify no exceptions are raised)
    # ------------------------------------------------------------------ #

    def test_rmse_comparison_plot_enabled(self):
        """plot_training_and_test_mse=True renders without raising an exception."""
        df = _make_df()
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=False,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=False,
                plot_forecast=False,
                plot_decomposition=False,
                plot_training_and_test_mse=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with RMSE comparison plot enabled: {e}")

    def test_rmse_comparison_plot_disabled(self):
        """plot_training_and_test_mse=False skips the chart without error."""
        df = _make_df()
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=False,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=False,
                plot_forecast=False,
                plot_decomposition=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with RMSE plot disabled: {e}")

    def test_custom_bar_colors(self):
        """Custom training_bar_color and test_bar_color are accepted without error."""
        df = _make_df()
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=False,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=False,
                plot_forecast=False,
                plot_decomposition=False,
                plot_training_and_test_mse=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")

    def test_performance_plot_smoke(self):
        """plot_model_performance=True renders without raising an exception."""
        df = _make_df()
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=False,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=True,
                plot_forecast=False,
                plot_decomposition=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with performance plot enabled: {e}")

    def test_forecast_plot_smoke(self):
        """plot_forecast=True renders without raising an exception."""
        df = _make_df()
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='simple',
                print_model_performance=False,
                print_parameter_summary=False,
                print_forecast_summary=False,
                plot_model_performance=False,
                plot_forecast=True,
                plot_decomposition=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with forecast plot enabled: {e}")

    def test_all_plots_enabled(self):
        """Enabling all standard plots raises no exception."""
        df = _make_df()
        try:
            CreateExponentialSmoothingModel(
                df, 'date', 'y',
                smoothing_type='double',
                print_model_performance=True,
                print_parameter_summary=True,
                print_forecast_summary=True,
                plot_model_performance=True,
                plot_forecast=True,
                plot_decomposition=False,
                plot_training_and_test_mse=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all plots enabled: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
