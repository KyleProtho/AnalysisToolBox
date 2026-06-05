import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must precede any pyplot import

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Pin to the local dev source tree (insert repo root so stdlib 'statistics' is not shadowed)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysistoolbox.predictive_analytics.CreateLogisticRegressionModel import CreateLogisticRegressionModel


class TestCreateLogisticRegressionModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(412)
        n = 300
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        y_cont = 3.0 * x1 + 1.5 * x2 + 2.0 + noise
        # Binary classification: above-median → 1
        y_clf = (y_cont > np.median(y_cont)).astype(int)
        self.df = pd.DataFrame({'x1': x1, 'x2': x2, 'y_clf': y_clf})
        self.outcome = 'y_clf'
        self.predictors = ['x1', 'x2']

    def tearDown(self):
        plt.clf()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # Return types
    # ------------------------------------------------------------------ #

    def test_returns_logistic_regression_when_not_scaled(self):
        """Returns a fitted LogisticRegression when scale_predictor_variables=False."""
        model = CreateLogisticRegressionModel(
            self.df, self.outcome, self.predictors,
            scale_predictor_variables=False,
            show_classification_plot=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsInstance(model, LogisticRegression)

    def test_returns_dict_when_scaled(self):
        """Returns a dict with 'model' and 'scaler' when scale_predictor_variables=True."""
        result = CreateLogisticRegressionModel(
            self.df, self.outcome, self.predictors,
            scale_predictor_variables=True,
            show_classification_plot=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('scaler', result)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

    def test_model_produces_one_prediction_per_row(self):
        """Returned model predicts one label per input row."""
        model = CreateLogisticRegressionModel(
            self.df, self.outcome, self.predictors,
            show_classification_plot=False,
            plot_training_and_test_mse=False,
        )
        preds = model.predict(self.df[self.predictors])
        self.assertEqual(len(preds), len(self.df))

    def test_predictions_are_binary_labels(self):
        """Predictions are restricted to the two class labels (0 and 1)."""
        model = CreateLogisticRegressionModel(
            self.df, self.outcome, self.predictors,
            show_classification_plot=False,
            plot_training_and_test_mse=False,
        )
        preds = model.predict(self.df[self.predictors])
        self.assertTrue(set(preds).issubset({0, 1}))

    # ------------------------------------------------------------------ #
    # Training and test accuracy output
    # ------------------------------------------------------------------ #

    def test_print_performance_includes_training_accuracy(self):
        """print_model_training_performance=True prints 'Training Accuracy:'."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                print_model_training_performance=True,
                show_classification_plot=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training Accuracy:', buf.getvalue())

    def test_print_performance_includes_test_accuracy(self):
        """print_model_training_performance=True prints 'Test Accuracy:'."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                print_model_training_performance=True,
                show_classification_plot=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test Accuracy:', buf.getvalue())

    def test_accuracy_values_are_valid(self):
        """Training Accuracy and Test Accuracy are floats in [0, 1]."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                print_model_training_performance=True,
                show_classification_plot=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_acc = float(next(l for l in lines if 'Training Accuracy:' in l).split(':')[1])
        test_acc  = float(next(l for l in lines if 'Test Accuracy:' in l).split(':')[1])
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 1.0)
        self.assertGreaterEqual(test_acc, 0.0)
        self.assertLessEqual(test_acc, 1.0)

    def test_print_performance_off_by_default(self):
        """With print_model_training_performance=False, no Training/Test Accuracy lines appear."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                print_model_training_performance=False,
                show_classification_plot=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertNotIn('Training Accuracy:', buf.getvalue())
        self.assertNotIn('Test Accuracy:', buf.getvalue())

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_single_predictor(self):
        """Function works with only one predictor variable."""
        model = CreateLogisticRegressionModel(
            self.df, self.outcome, ['x1'],
            show_classification_plot=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsNotNone(model)

    def test_handles_missing_values(self):
        """Rows with NaN are dropped before training."""
        df_nan = self.df.copy()
        df_nan.loc[:5, 'x1'] = np.nan
        model = CreateLogisticRegressionModel(
            df_nan, self.outcome, self.predictors,
            show_classification_plot=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsNotNone(model)

    # ------------------------------------------------------------------ #
    # Plot smoke tests (verify no exceptions are raised)
    # ------------------------------------------------------------------ #

    def test_all_plots_enabled(self):
        """Enabling all plots raises no exception."""
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                show_classification_plot=True,
                plot_training_and_test_mse=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all plots enabled: {e}")

    def test_accuracy_comparison_plot_disabled(self):
        """plot_training_and_test_mse=False skips the comparison chart without error."""
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                show_classification_plot=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with accuracy plot disabled: {e}")

    def test_custom_bar_colors(self):
        """Custom training_bar_color and test_bar_color are accepted without error."""
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                show_classification_plot=False,
                plot_training_and_test_mse=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")

    def test_confusion_matrix_plot_smoke(self):
        """show_classification_plot=True renders without raising an exception."""
        try:
            CreateLogisticRegressionModel(
                self.df, self.outcome, self.predictors,
                show_classification_plot=True,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with confusion matrix plot enabled: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
