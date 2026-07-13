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
from analysistoolbox.predictive_analytics.CreateNeuralNetwork_SingleOutcome import CreateNeuralNetwork_SingleOutcome


def _make_df(seed=412, n=200):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    noise = rng.standard_normal(n) * 0.5
    y_reg = 3.0 * x1 + 1.5 * x2 + 2.0 + noise
    y_clf = (y_reg > np.median(y_reg)).astype(int)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y_reg': y_reg, 'y_clf': y_clf})


class TestCreateNeuralNetworkSingleOutcome(unittest.TestCase):

    def tearDown(self):
        plt.clf()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # Return types
    # ------------------------------------------------------------------ #

    def test_returns_model_when_not_scaled(self):
        """Returns a Keras Model when scale_predictor_variables=False."""
        import tensorflow as tf
        df = _make_df()
        result = CreateNeuralNetwork_SingleOutcome(
            df, 'y_reg', ['x1', 'x2'],
            number_of_hidden_layers=1,
            is_outcome_categorical=False,
            scale_predictor_variables=False,
            number_of_steps_gradient_descent=5,
            plot_loss=False,
            plot_model_test_performance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(result, tf.keras.Model)

    def test_returns_dict_when_scaled(self):
        """Returns a dict with 'model' and 'scaler' when scale_predictor_variables=True."""
        df = _make_df()
        result = CreateNeuralNetwork_SingleOutcome(
            df, 'y_reg', ['x1', 'x2'],
            number_of_hidden_layers=1,
            is_outcome_categorical=False,
            scale_predictor_variables=True,
            number_of_steps_gradient_descent=5,
            plot_loss=False,
            plot_model_test_performance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('scaler', result)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

    def test_regression_model_produces_predictions(self):
        """Returned regression model's predict() produces one value per row."""
        df = _make_df()
        result = CreateNeuralNetwork_SingleOutcome(
            df, 'y_reg', ['x1', 'x2'],
            number_of_hidden_layers=1,
            is_outcome_categorical=False,
            scale_predictor_variables=False,
            number_of_steps_gradient_descent=5,
            plot_loss=False,
            plot_model_test_performance=False,
            plot_training_and_test_performance=False,
        )
        preds = result.predict(df[['x1', 'x2']].values)
        self.assertEqual(len(preds), len(df))

    def test_classifier_model_produces_predictions(self):
        """Returned classifier's predict() produces one value per row."""
        df = _make_df()
        result = CreateNeuralNetwork_SingleOutcome(
            df, 'y_clf', ['x1', 'x2'],
            number_of_hidden_layers=1,
            is_outcome_categorical=True,
            scale_predictor_variables=False,
            number_of_steps_gradient_descent=5,
            plot_loss=False,
            plot_model_test_performance=False,
            plot_training_and_test_performance=False,
        )
        preds = result.predict(df[['x1', 'x2']].values)
        self.assertEqual(len(preds), len(df))

    # ------------------------------------------------------------------ #
    # Training and test MSE output — regression
    # ------------------------------------------------------------------ #

    def test_print_performance_regression_includes_training_mse(self):
        """print_model_training_performance=True prints 'Training MSE:' for regression."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_reg', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=False,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                print_model_training_performance=True,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training MSE:', buf.getvalue())

    def test_print_performance_regression_includes_test_mse(self):
        """print_model_training_performance=True prints 'Test MSE:' for regression."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_reg', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=False,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                print_model_training_performance=True,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test MSE:', buf.getvalue())

    def test_regression_mse_values_are_positive_finite(self):
        """Printed Training MSE and Test MSE are positive, finite floats."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_reg', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=False,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                print_model_training_performance=True,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_mse = float(next(l for l in lines if 'Training MSE:' in l).split(':')[1])
        test_mse  = float(next(l for l in lines if 'Test MSE:' in l).split(':')[1])
        self.assertGreaterEqual(train_mse, 0)
        self.assertGreater(test_mse, 0)
        self.assertTrue(np.isfinite(train_mse))
        self.assertTrue(np.isfinite(test_mse))

    # ------------------------------------------------------------------ #
    # Training and test error rate output — classification
    # ------------------------------------------------------------------ #

    def test_print_performance_classification_includes_training_error_rate(self):
        """print_model_training_performance=True prints 'Training Error Rate:' for classification."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_clf', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=True,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                print_model_training_performance=True,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training Error Rate:', buf.getvalue())

    def test_print_performance_classification_includes_test_error_rate(self):
        """print_model_training_performance=True prints 'Test Error Rate:' for classification."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_clf', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=True,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                print_model_training_performance=True,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test Error Rate:', buf.getvalue())

    def test_classification_error_rate_values_are_valid(self):
        """Training Error Rate and Test Error Rate are floats in [0, 1]."""
        df = _make_df()
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_clf', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=True,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                print_model_training_performance=True,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_error_rate = float(next(l for l in lines if 'Training Error Rate:' in l).split(':')[1])
        test_error_rate  = float(next(l for l in lines if 'Test Error Rate:' in l).split(':')[1])
        self.assertGreaterEqual(train_error_rate, 0.0)
        self.assertLessEqual(train_error_rate, 1.0)
        self.assertGreaterEqual(test_error_rate, 0.0)
        self.assertLessEqual(test_error_rate, 1.0)

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_single_predictor(self):
        """Function works with only one predictor variable."""
        df = _make_df()
        result = CreateNeuralNetwork_SingleOutcome(
            df, 'y_reg', ['x1'],
            number_of_hidden_layers=1,
            is_outcome_categorical=False,
            scale_predictor_variables=False,
            number_of_steps_gradient_descent=5,
            plot_loss=False,
            plot_model_test_performance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(result)

    def test_multiple_hidden_layers(self):
        """Function trains without error when number_of_hidden_layers=3."""
        df = _make_df()
        result = CreateNeuralNetwork_SingleOutcome(
            df, 'y_reg', ['x1', 'x2'],
            number_of_hidden_layers=3,
            is_outcome_categorical=False,
            scale_predictor_variables=False,
            number_of_steps_gradient_descent=5,
            plot_loss=False,
            plot_model_test_performance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(result)

    # ------------------------------------------------------------------ #
    # Plot smoke tests (verify no exceptions are raised)
    # ------------------------------------------------------------------ #

    def test_all_plots_enabled_regression(self):
        """Enabling all plots for regression raises no exception."""
        df = _make_df()
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_reg', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=False,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                plot_loss=True,
                plot_model_test_performance=True,
                plot_training_and_test_performance=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all regression plots enabled: {e}")

    def test_all_plots_enabled_classification(self):
        """Enabling all plots for classification raises no exception."""
        df = _make_df()
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_clf', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=True,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                plot_loss=True,
                plot_model_test_performance=True,
                plot_training_and_test_performance=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all classification plots enabled: {e}")

    def test_performance_comparison_plot_disabled(self):
        """plot_training_and_test_performance=False skips the comparison chart without error."""
        df = _make_df()
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_reg', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=False,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with performance plot disabled: {e}")

    def test_custom_performance_bar_colors(self):
        """Custom training_bar_color and test_bar_color are accepted without error."""
        df = _make_df()
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_reg', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=False,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                plot_loss=False,
                plot_model_test_performance=False,
                plot_training_and_test_performance=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")

    def test_loss_plot_smoke(self):
        """plot_loss=True renders without raising an exception."""
        df = _make_df()
        try:
            CreateNeuralNetwork_SingleOutcome(
                df, 'y_reg', ['x1', 'x2'],
                number_of_hidden_layers=1,
                is_outcome_categorical=False,
                scale_predictor_variables=False,
                number_of_steps_gradient_descent=5,
                plot_loss=True,
                plot_model_test_performance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with loss plot enabled: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
