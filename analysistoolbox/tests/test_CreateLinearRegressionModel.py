import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must precede any pyplot import

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Pin to the local dev source tree so the installed PyPI version is not used
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'analysistoolbox'))
from analysistoolbox.predictive_analytics.CreateLinearRegressionModel import CreateLinearRegressionModel


class TestCreateLinearRegressionModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(412)
        n = 300
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        # Known relationship: y ≈ 3*x1 + 1.5*x2 + 2
        y = 3.0 * x1 + 1.5 * x2 + 2.0 + noise
        self.df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
        self.outcome = 'y'
        self.predictors = ['x1', 'x2']

    def tearDown(self):
        plt.clf()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # Return type
    # ------------------------------------------------------------------ #

    def test_returns_linear_regression_model(self):
        """Returns a fitted LinearRegression when scale_variables=False."""
        model = CreateLinearRegressionModel(
            self.df, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, LinearRegression)

    def test_returns_pipeline_when_scaled(self):
        """Returns a sklearn Pipeline when scale_variables=True."""
        model = CreateLinearRegressionModel(
            self.df, self.outcome, self.predictors,
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, Pipeline)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

    def test_model_produces_one_prediction_per_row(self):
        """Returned model produces exactly one prediction per input row."""
        model = CreateLinearRegressionModel(
            self.df, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        preds = model.predict(self.df[self.predictors])
        self.assertEqual(len(preds), len(self.df))

    def test_coefficients_have_correct_sign(self):
        """Beta coefficients should be positive (data is y ≈ 3*x1 + 1.5*x2 + noise)."""
        model = CreateLinearRegressionModel(
            self.df, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        for coef in model.coef_:
            self.assertGreater(coef, 0)

    # ------------------------------------------------------------------ #
    # Training and test MSE output
    # ------------------------------------------------------------------ #

    def test_print_performance_includes_training_mse(self):
        """print_model_training_performance=True prints a 'Training MSE:' line."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateLinearRegressionModel(
                self.df, self.outcome, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training MSE:', buf.getvalue())

    def test_print_performance_includes_test_mse(self):
        """print_model_training_performance=True prints a 'Test MSE:' line."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateLinearRegressionModel(
                self.df, self.outcome, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test MSE:', buf.getvalue())

    def test_both_mses_are_positive_finite_numbers(self):
        """Printed Training MSE and Test MSE values are positive, finite floats."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateLinearRegressionModel(
                self.df, self.outcome, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_mse = float(next(l for l in lines if 'Training MSE:' in l).split(':')[1])
        test_mse  = float(next(l for l in lines if 'Test MSE:' in l).split(':')[1])
        self.assertGreater(train_mse, 0)
        self.assertGreater(test_mse, 0)
        self.assertTrue(np.isfinite(train_mse))
        self.assertTrue(np.isfinite(test_mse))

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_test_size_zero_uses_full_dataset(self):
        """test_size=0 trains and evaluates on the full dataset without error."""
        model = CreateLinearRegressionModel(
            self.df, self.outcome, self.predictors,
            test_size=0,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    def test_handles_missing_values(self):
        """Rows with NaN values are silently dropped before training."""
        df_nan = self.df.copy()
        df_nan.loc[:5, 'x1'] = np.nan
        model = CreateLinearRegressionModel(
            df_nan, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    def test_single_predictor(self):
        """Function works correctly with only one predictor variable."""
        model = CreateLinearRegressionModel(
            self.df, self.outcome, ['x1'],
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    # ------------------------------------------------------------------ #
    # Plot smoke tests (verify no exceptions are raised)
    # ------------------------------------------------------------------ #

    def test_all_plots_enabled(self):
        """Enabling all three plots does not raise an exception."""
        try:
            CreateLinearRegressionModel(
                self.df, self.outcome, self.predictors,
                plot_model_test_performance=True,
                plot_feature_importance=True,
                plot_training_and_test_performance=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all plots enabled: {e}")

    def test_performance_comparison_plot_disabled(self):
        """plot_training_and_test_performance=False skips the MSE chart without error."""
        try:
            CreateLinearRegressionModel(
                self.df, self.outcome, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with MSE plot disabled: {e}")

    def test_custom_performance_bar_colors(self):
        """Custom training_bar_color and test_bar_color arguments are accepted."""
        try:
            CreateLinearRegressionModel(
                self.df, self.outcome, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
