import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must precede any pyplot import

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.pipeline import Pipeline

# Pin to the local dev source tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysistoolbox.predictive_analytics.CreateKNearestNeighborsModel import CreateKNearestNeighborsModel


class TestCreateKNearestNeighborsModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(412)
        n = 300
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        y_reg = 3.0 * x1 + 1.5 * x2 + 2.0 + noise
        y_clf = (y_reg > np.median(y_reg)).astype(int)
        self.df = pd.DataFrame({'x1': x1, 'x2': x2, 'y_reg': y_reg, 'y_clf': y_clf})
        self.outcome_reg = 'y_reg'
        self.outcome_clf = 'y_clf'
        self.predictors = ['x1', 'x2']

    def tearDown(self):
        plt.clf()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # Return types
    # ------------------------------------------------------------------ #

    def test_returns_knn_regressor_when_no_scaling(self):
        """Returns a fitted KNeighborsRegressor when scale_variables=False."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, self.predictors,
            model_type='regressor',
            scale_variables=False,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, KNeighborsRegressor)

    def test_returns_knn_classifier_when_no_scaling(self):
        """Returns a fitted KNeighborsClassifier when scale_variables=False."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_clf, self.predictors,
            model_type='classifier',
            scale_variables=False,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, KNeighborsClassifier)

    def test_returns_pipeline_when_scale_variables_true(self):
        """Returns a Pipeline when scale_variables=True."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, self.predictors,
            model_type='regressor',
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, Pipeline)

    def test_pipeline_contains_knn_regressor(self):
        """Pipeline final estimator is a KNeighborsRegressor."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, self.predictors,
            model_type='regressor',
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model[-1], KNeighborsRegressor)

    def test_pipeline_contains_knn_classifier(self):
        """Pipeline final estimator is a KNeighborsClassifier."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_clf, self.predictors,
            model_type='classifier',
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model[-1], KNeighborsClassifier)

    def test_returns_radius_regressor(self):
        """Returns a RadiusNeighborsRegressor (or Pipeline wrapping one) for radius_regressor."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, self.predictors,
            model_type='radius_regressor',
            radius=5.0,
            scale_variables=False,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, RadiusNeighborsRegressor)

    def test_returns_radius_classifier(self):
        """Returns a RadiusNeighborsClassifier (or Pipeline wrapping one) for radius_classifier."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_clf, self.predictors,
            model_type='radius_classifier',
            radius=5.0,
            scale_variables=False,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, RadiusNeighborsClassifier)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

    def test_regressor_produces_one_prediction_per_row(self):
        """Returned regression model predicts one value per input row."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, self.predictors,
            model_type='regressor',
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        preds = model.predict(self.df[self.predictors])
        self.assertEqual(len(preds), len(self.df))

    def test_classifier_produces_one_prediction_per_row(self):
        """Returned classifier produces one label per input row."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_clf, self.predictors,
            model_type='classifier',
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        preds = model.predict(self.df[self.predictors])
        self.assertEqual(len(preds), len(self.df))

    # ------------------------------------------------------------------ #
    # Model parameter validation
    # ------------------------------------------------------------------ #

    def test_raises_value_error_for_invalid_model_type(self):
        """Raises ValueError when model_type is not a recognized string."""
        with self.assertRaises(ValueError):
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                model_type='invalid_type',
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )

    def test_custom_n_neighbors_accepted(self):
        """Custom n_neighbors value is accepted without error."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, self.predictors,
            n_neighbors=3,
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    def test_distance_weights_accepted(self):
        """weights='distance' is accepted without error."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, self.predictors,
            weights='distance',
            scale_variables=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    # ------------------------------------------------------------------ #
    # Printed performance output — regression
    # ------------------------------------------------------------------ #

    def test_print_performance_regression_includes_training_mse(self):
        """print_model_training_performance=True prints 'Training MSE:' for regression."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training MSE:', buf.getvalue())

    def test_print_performance_regression_includes_test_mse(self):
        """print_model_training_performance=True prints 'Test MSE:' for regression."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test MSE:', buf.getvalue())

    def test_regression_mse_values_are_positive_finite(self):
        """Printed Training MSE and Test MSE are non-negative finite floats."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_mse = float(next(l for l in lines if 'Training MSE:' in l).split(':')[1])
        test_mse = float(next(l for l in lines if 'Test MSE:' in l).split(':')[1])
        self.assertGreaterEqual(train_mse, 0)
        self.assertGreater(test_mse, 0)
        self.assertTrue(np.isfinite(train_mse))
        self.assertTrue(np.isfinite(test_mse))

    # ------------------------------------------------------------------ #
    # Printed performance output — classification
    # ------------------------------------------------------------------ #

    def test_print_performance_classification_includes_training_error_rate(self):
        """print_model_training_performance=True prints 'Training Error Rate:' for classification."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_clf, self.predictors,
                model_type='classifier',
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training Error Rate:', buf.getvalue())

    def test_print_performance_classification_includes_test_error_rate(self):
        """print_model_training_performance=True prints 'Test Error Rate:' for classification."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_clf, self.predictors,
                model_type='classifier',
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test Error Rate:', buf.getvalue())

    def test_print_performance_classification_includes_classification_report(self):
        """Classification Report is printed alongside error rate metrics."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_clf, self.predictors,
                model_type='classifier',
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Classification Report:', buf.getvalue())

    def test_classification_error_rate_values_are_valid(self):
        """Training Error Rate and Test Error Rate are floats in [0, 1]."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_clf, self.predictors,
                model_type='classifier',
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_err = float(next(l for l in lines if 'Training Error Rate:' in l).split(':')[1])
        test_err = float(next(l for l in lines if 'Test Error Rate:' in l).split(':')[1])
        self.assertGreaterEqual(train_err, 0.0)
        self.assertLessEqual(train_err, 1.0)
        self.assertGreaterEqual(test_err, 0.0)
        self.assertLessEqual(test_err, 1.0)

    # ------------------------------------------------------------------ #
    # Curse of dimensionality warning
    # ------------------------------------------------------------------ #

    def test_warning_printed_for_five_predictors(self):
        """A warning is printed when 5 or more predictors are provided."""
        df_wide = self.df.copy()
        df_wide['x3'] = np.random.randn(len(df_wide))
        df_wide['x4'] = np.random.randn(len(df_wide))
        df_wide['x5'] = np.random.randn(len(df_wide))
        five_predictors = ['x1', 'x2', 'x3', 'x4', 'x5']

        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                df_wide, self.outcome_reg, five_predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('WARNING', buf.getvalue())

    def test_no_warning_for_four_predictors(self):
        """No dimensionality warning is printed when 4 or fewer predictors are provided."""
        df_wide = self.df.copy()
        df_wide['x3'] = np.random.randn(len(df_wide))
        df_wide['x4'] = np.random.randn(len(df_wide))
        four_predictors = ['x1', 'x2', 'x3', 'x4']

        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateKNearestNeighborsModel(
                df_wide, self.outcome_reg, four_predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertNotIn('WARNING', buf.getvalue())

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_single_predictor(self):
        """Function works with only one predictor variable."""
        model = CreateKNearestNeighborsModel(
            self.df, self.outcome_reg, ['x1'],
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    def test_handles_missing_values(self):
        """Rows with NaN values are dropped before training."""
        df_nan = self.df.copy()
        df_nan.loc[:5, 'x1'] = np.nan
        model = CreateKNearestNeighborsModel(
            df_nan, self.outcome_reg, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    # ------------------------------------------------------------------ #
    # Plot smoke tests
    # ------------------------------------------------------------------ #

    def test_all_plots_enabled_regression(self):
        """Enabling all plots for regression raises no exception."""
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                model_type='regressor',
                plot_model_test_performance=True,
                plot_feature_importance=True,
                plot_training_and_test_performance=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all regression plots enabled: {e}")

    def test_all_plots_enabled_classification(self):
        """Enabling all plots for classification raises no exception."""
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_clf, self.predictors,
                model_type='classifier',
                plot_model_test_performance=True,
                plot_feature_importance=True,
                plot_training_and_test_performance=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all classification plots enabled: {e}")

    def test_all_plots_disabled(self):
        """Disabling all plots raises no exception."""
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all plots disabled: {e}")

    def test_custom_performance_bar_colors(self):
        """Custom training_bar_color and test_bar_color are accepted without error."""
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")

    def test_radius_regressor_smoke(self):
        """radius_regressor model type trains and predicts without error."""
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_reg, self.predictors,
                model_type='radius_regressor',
                radius=5.0,
                scale_variables=False,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with radius_regressor: {e}")

    def test_radius_classifier_smoke(self):
        """radius_classifier model type trains and predicts without error."""
        try:
            CreateKNearestNeighborsModel(
                self.df, self.outcome_clf, self.predictors,
                model_type='radius_classifier',
                radius=5.0,
                scale_variables=False,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with radius_classifier: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
