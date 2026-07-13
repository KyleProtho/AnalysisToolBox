import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must precede any pyplot import

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB

# Pin to the local dev source tree (insert repo root so stdlib 'statistics' is not shadowed)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analysistoolbox.predictive_analytics.CreateBayesClassifier import CreateBayesClassifier


def _make_continuous_df(n=300, seed=412):
    """Binary-outcome DataFrame with two continuous predictors."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y_cont = 3.0 * x1 + 1.5 * x2 + rng.standard_normal(n) * 0.5
    y_clf = (y_cont > np.median(y_cont)).astype(int)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y_clf': y_clf})


def _make_count_df(n=300, seed=412):
    """Binary-outcome DataFrame with non-negative integer predictors."""
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 20, size=n)
    x2 = rng.integers(0, 20, size=n)
    y_clf = (x1 + x2 > np.median(x1 + x2)).astype(int)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y_clf': y_clf})


def _make_binary_df(n=300, seed=412):
    """Binary-outcome DataFrame with binary predictors."""
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, size=n)
    x2 = rng.integers(0, 2, size=n)
    y_clf = ((x1 + x2) > 0).astype(int)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y_clf': y_clf})


class TestCreateBayesClassifier(unittest.TestCase):

    def setUp(self):
        self.df_continuous = _make_continuous_df()
        self.df_count = _make_count_df()
        self.df_binary = _make_binary_df()
        self.outcome = 'y_clf'
        self.predictors = ['x1', 'x2']

    def tearDown(self):
        plt.clf()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # Return types
    # ------------------------------------------------------------------ #

    def test_returns_gaussian_nb_by_default(self):
        """Default naive_bayes_type='gaussian' returns a fitted GaussianNB."""
        model = CreateBayesClassifier(
            self.df_continuous, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, GaussianNB)

    def test_returns_model_object(self):
        """Return value is not None and has a predict method."""
        model = CreateBayesClassifier(
            self.df_continuous, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    # ------------------------------------------------------------------ #
    # NB type variants
    # ------------------------------------------------------------------ #

    def test_multinomial_nb_trains_without_error(self):
        """naive_bayes_type='multinomial' trains on count data without error."""
        try:
            CreateBayesClassifier(
                self.df_count, self.outcome, self.predictors,
                naive_bayes_type='multinomial',
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"MultinomialNB raised an unexpected exception: {e}")

    def test_multinomial_nb_returns_correct_type(self):
        """naive_bayes_type='multinomial' returns a MultinomialNB instance."""
        model = CreateBayesClassifier(
            self.df_count, self.outcome, self.predictors,
            naive_bayes_type='multinomial',
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, MultinomialNB)

    def test_bernoulli_nb_trains_without_error(self):
        """naive_bayes_type='bernoulli' trains on binary data without error."""
        try:
            CreateBayesClassifier(
                self.df_binary, self.outcome, self.predictors,
                naive_bayes_type='bernoulli',
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"BernoulliNB raised an unexpected exception: {e}")

    def test_bernoulli_nb_returns_correct_type(self):
        """naive_bayes_type='bernoulli' returns a BernoulliNB instance."""
        model = CreateBayesClassifier(
            self.df_binary, self.outcome, self.predictors,
            naive_bayes_type='bernoulli',
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, BernoulliNB)

    def test_complement_nb_trains_without_error(self):
        """naive_bayes_type='complement' trains on count data without error."""
        try:
            CreateBayesClassifier(
                self.df_count, self.outcome, self.predictors,
                naive_bayes_type='complement',
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"ComplementNB raised an unexpected exception: {e}")

    def test_complement_nb_returns_correct_type(self):
        """naive_bayes_type='complement' returns a ComplementNB instance."""
        model = CreateBayesClassifier(
            self.df_count, self.outcome, self.predictors,
            naive_bayes_type='complement',
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, ComplementNB)

    def test_categorical_nb_trains_without_error(self):
        """naive_bayes_type='categorical' trains on non-negative integer data without error."""
        try:
            CreateBayesClassifier(
                self.df_count, self.outcome, self.predictors,
                naive_bayes_type='categorical',
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"CategoricalNB raised an unexpected exception: {e}")

    def test_categorical_nb_returns_correct_type(self):
        """naive_bayes_type='categorical' returns a CategoricalNB instance."""
        model = CreateBayesClassifier(
            self.df_count, self.outcome, self.predictors,
            naive_bayes_type='categorical',
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsInstance(model, CategoricalNB)

    def test_invalid_type_raises_value_error(self):
        """An unrecognized naive_bayes_type raises ValueError."""
        with self.assertRaises(ValueError):
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                naive_bayes_type='unknown',
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

    def test_model_produces_one_prediction_per_row(self):
        """Returned model predicts one label per input row."""
        model = CreateBayesClassifier(
            self.df_continuous, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        preds = model.predict(self.df_continuous[self.predictors])
        self.assertEqual(len(preds), len(self.df_continuous))

    def test_predictions_are_valid_class_labels(self):
        """Predictions are restricted to class labels present in the training data."""
        model = CreateBayesClassifier(
            self.df_continuous, self.outcome, self.predictors,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        preds = model.predict(self.df_continuous[self.predictors])
        valid_labels = set(self.df_continuous[self.outcome].unique())
        self.assertTrue(set(preds).issubset(valid_labels))

    # ------------------------------------------------------------------ #
    # Training and test error rate output
    # ------------------------------------------------------------------ #

    def test_print_performance_includes_training_error_rate(self):
        """print_model_training_performance=True prints 'Training Error Rate:'."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Training Error Rate:', buf.getvalue())

    def test_print_performance_includes_test_error_rate(self):
        """print_model_training_performance=True prints 'Test Error Rate:'."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn('Test Error Rate:', buf.getvalue())

    def test_error_rate_values_are_valid(self):
        """Training Error Rate and Test Error Rate are floats in [0, 1]."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                print_model_training_performance=True,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        lines = buf.getvalue().splitlines()
        train_error_rate = float(next(l for l in lines if 'Training Error Rate:' in l).split(':')[1])
        test_error_rate = float(next(l for l in lines if 'Test Error Rate:' in l).split(':')[1])
        self.assertGreaterEqual(train_error_rate, 0.0)
        self.assertLessEqual(train_error_rate, 1.0)
        self.assertGreaterEqual(test_error_rate, 0.0)
        self.assertLessEqual(test_error_rate, 1.0)

    def test_print_performance_off_by_default(self):
        """With print_model_training_performance=False, no Error Rate lines appear."""
        buf = io.StringIO()
        sys.stdout = buf
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                print_model_training_performance=False,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        finally:
            sys.stdout = sys.__stdout__
        self.assertNotIn('Training Error Rate:', buf.getvalue())
        self.assertNotIn('Test Error Rate:', buf.getvalue())

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_single_predictor(self):
        """Function works with only one predictor variable."""
        model = CreateBayesClassifier(
            self.df_continuous, self.outcome, ['x1'],
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    def test_filter_nulls_drops_nan_rows(self):
        """Rows with NaN are dropped before training when filter_nulls=True."""
        df_nan = self.df_continuous.copy()
        df_nan.loc[:5, 'x1'] = np.nan
        model = CreateBayesClassifier(
            df_nan, self.outcome, self.predictors,
            filter_nulls=True,
            plot_model_test_performance=False,
            plot_feature_importance=False,
            plot_training_and_test_performance=False,
        )
        self.assertIsNotNone(model)

    # ------------------------------------------------------------------ #
    # Plot smoke tests (verify no exceptions are raised)
    # ------------------------------------------------------------------ #

    def test_all_plots_enabled(self):
        """Enabling all plots raises no exception."""
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                plot_model_test_performance=True,
                plot_feature_importance=True,
                plot_training_and_test_performance=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all plots enabled: {e}")

    def test_confusion_matrix_plot_smoke(self):
        """plot_model_test_performance=True renders without raising an exception."""
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                plot_model_test_performance=True,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with confusion matrix enabled: {e}")

    def test_feature_discriminability_plot_smoke(self):
        """plot_feature_importance=True renders without raising an exception."""
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=True,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with feature discriminability plot enabled: {e}")

    def test_performance_comparison_plot_smoke(self):
        """plot_training_and_test_performance=True renders without raising an exception."""
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=True,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with performance comparison plot enabled: {e}")

    def test_all_plots_disabled(self):
        """Disabling all plots runs without error."""
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception with all plots disabled: {e}")

    def test_custom_bar_colors(self):
        """Custom training_bar_color and test_bar_color are accepted without error."""
        try:
            CreateBayesClassifier(
                self.df_continuous, self.outcome, self.predictors,
                plot_model_test_performance=False,
                plot_feature_importance=False,
                plot_training_and_test_performance=True,
                training_bar_color='green',
                test_bar_color='orange',
            )
        except Exception as e:
            self.fail(f"Unexpected exception with custom bar colors: {e}")

    def test_feature_discriminability_multinomial(self):
        """Feature discriminability plot works for non-Gaussian NB variants."""
        try:
            CreateBayesClassifier(
                self.df_count, self.outcome, self.predictors,
                naive_bayes_type='multinomial',
                plot_model_test_performance=False,
                plot_feature_importance=True,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception for MultinomialNB discriminability plot: {e}")

    def test_feature_discriminability_categorical(self):
        """Feature discriminability plot works for CategoricalNB."""
        try:
            CreateBayesClassifier(
                self.df_count, self.outcome, self.predictors,
                naive_bayes_type='categorical',
                plot_model_test_performance=False,
                plot_feature_importance=True,
                plot_training_and_test_performance=False,
            )
        except Exception as e:
            self.fail(f"Unexpected exception for CategoricalNB discriminability plot: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
