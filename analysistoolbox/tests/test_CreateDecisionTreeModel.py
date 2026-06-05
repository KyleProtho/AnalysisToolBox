import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must precede any pyplot import

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Pin to the local dev source tree (insert repo root so stdlib 'statistics' is not shadowed)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysistoolbox.predictive_analytics.CreateDecisionTreeModel import CreateDecisionTreeModel


class TestCreateDecisionTreeModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(412)
        n = 300
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        noise = np.random.randn(n) * 0.5
        y_reg = 3.0 * x1 + 1.5 * x2 + 2.0 + noise
        # Binary classification: above-median → 1
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

    def test_returns_decision_tree_regressor(self):
        """Returns a fitted DecisionTreeRegressor for a continuous outcome."""
        model = CreateDecisionTreeModel(
            self.df, self.outcome_reg, self.predictors,
            is_outcome_categorical=False,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsInstance(model, DecisionTreeRegressor)

    def test_returns_decision_tree_classifier(self):
        """Returns a fitted DecisionTreeClassifier for a categorical outcome."""
        model = CreateDecisionTreeModel(
            self.df, self.outcome_clf, self.predictors,
            is_outcome_categorical=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsInstance(model, DecisionTreeClassifier)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

    def test_regression_model_produces_one_prediction_per_row(self):
        """Returned regression model predicts one value per input row."""
        model = CreateDecisionTreeModel(
            self.df, self.outcome_reg, self.predictors,
            is_outcome_categorical=False,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_mse=False,
        )
        preds = model.predict(self.df[self.predictors])
        self.assertEqual(len(preds), len(self.df))

    def test_classifier_model_produces_one_prediction_per_row(self):
        """Returned classifier produces one label per input row."""
        model = CreateDecisionTreeModel(
            self.df, self.outcome_clf, self.predictors,
            is_outcome_categorical=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_mse=False,
        )
        preds = model.predict(self.df[self.predictors])
        self.assertEqual(len(preds), len(self.df))

    # ------------------------------------------------------------------ #
    # Training and test MSE output — regression
    # ------------------------------------------------------------------ #

    def test_print_performance_regression_includes_training_mse(self):
        """print_model_training_performance=True prints 'Training MSE:' for regression."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training MSE:', buf.getvalue())

    def test_print_performance_regression_includes_test_mse(self):
        """print_model_training_performance=True prints 'Test MSE:' for regression."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test MSE:', buf.getvalue())

    def test_regression_mse_values_are_positive_finite(self):
        """Printed Training MSE and Test MSE are positive, finite floats."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
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
    # Training and test accuracy output — classification
    # ------------------------------------------------------------------ #

    def test_print_performance_classification_includes_training_accuracy(self):
        """print_model_training_performance=True prints 'Training Accuracy:' for classification."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_clf, self.predictors,
                is_outcome_categorical=True,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training Accuracy:', buf.getvalue())

    def test_print_performance_classification_includes_test_accuracy(self):
        """print_model_training_performance=True prints 'Test Accuracy:' for classification."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_clf, self.predictors,
                is_outcome_categorical=True,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test Accuracy:', buf.getvalue())

    def test_classification_accuracy_values_are_valid(self):
        """Training Accuracy and Test Accuracy are floats in [0, 1]."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_clf, self.predictors,
                is_outcome_categorical=True,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
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

    def test_print_performance_classification_includes_classification_report(self):
        """Classification Report is printed alongside accuracy metrics."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_clf, self.predictors,
                is_outcome_categorical=True,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Classification Report:', buf.getvalue())

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_single_predictor(self):
        """Function works with only one predictor variable."""
        model = CreateDecisionTreeModel(
            self.df, self.outcome_reg, ['x1'],
            is_outcome_categorical=False,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsNotNone(model)

    def test_max_depth_limits_tree(self):
        """maximum_depth=1 produces a single-split stump."""
        model = CreateDecisionTreeModel(
            self.df, self.outcome_reg, self.predictors,
            is_outcome_categorical=False,
            maximum_depth=1,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_mse=False,
        )
        self.assertEqual(model.get_depth(), 1)

    def test_handles_missing_values(self):
        """Rows with NaN are silently dropped before training."""
        df_nan = self.df.copy()
        df_nan.loc[:5, 'x1'] = np.nan
        model = CreateDecisionTreeModel(
            df_nan, self.outcome_reg, self.predictors,
            is_outcome_categorical=False,
            filter_nulls=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_mse=False,
        )
        self.assertIsNotNone(model)

    # ------------------------------------------------------------------ #
    # Plot smoke tests (verify no exceptions are raised)
    # ------------------------------------------------------------------ #

    def test_all_plots_enabled_regression(self):
        """Enabling all plots for regression raises no exception."""
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                plot_model_test_performance=True,
                plot_feature_importance=True,
                plot_training_and_test_mse=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all regression plots enabled: {e}")

    def test_all_plots_enabled_classification(self):
        """Enabling all plots for classification raises no exception."""
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_clf, self.predictors,
                is_outcome_categorical=True,
                plot_model_test_performance=True,
                plot_feature_importance=True,
                plot_training_and_test_mse=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all classification plots enabled: {e}")

    def test_mse_comparison_plot_disabled(self):
        """plot_training_and_test_mse=False skips the comparison chart without error."""
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with MSE plot disabled: {e}")

    def test_custom_mse_bar_colors(self):
        """Custom training_bar_color and test_bar_color are accepted without error."""
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")

    def test_decision_tree_plot_smoke(self):
        """plot_decision_tree=True renders without raising an exception."""
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                maximum_depth=2,
                plot_decision_tree=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with decision tree plot enabled: {e}")

    def test_print_decision_rules_smoke(self):
        """print_decision_rules=True prints output without raising an exception."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateDecisionTreeModel(
                self.df, self.outcome_reg, self.predictors,
                is_outcome_categorical=False,
                maximum_depth=2,
                print_decision_rules=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_mse=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with print_decision_rules=True: {e}")
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('|', buf.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
