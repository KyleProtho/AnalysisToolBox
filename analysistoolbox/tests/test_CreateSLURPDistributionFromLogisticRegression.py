# Load packages
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from unittest.mock import patch
from analysistoolbox.simulations import CreateSLURPDistributionFromLogisticRegression

class TestCreateSLURPDistributionFromLogisticRegression(unittest.TestCase):

    def setUp(self):
        # Fit a small logistic regression model to use across tests
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        logit_p = 0.5 + 1.5 * x1 - 0.8 * x2
        p = 1 / (1 + np.exp(-logit_p))
        y = rng.binomial(1, p)

        self.predictors = sm.add_constant(pd.DataFrame({'x1': x1, 'x2': x2}))
        self.logit_model = sm.Logit(y, self.predictors).fit(disp=0)

        # A non-logistic model, for negative testing
        y_continuous = logit_p + rng.normal(0, 0.1, n)
        self.ols_model = sm.OLS(y_continuous, self.predictors).fit()

    def test_returns_dataframe_by_default(self):
        result = CreateSLURPDistributionFromLogisticRegression(
            logistic_regression_model=self.logit_model,
            list_of_prediction_values=[0.5, -0.5],
            number_of_trials=200,
            plot_simulation_results=False
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 200)
        self.assertEqual(result.columns.tolist(), ['y_probability'])

    def test_returns_array_when_requested(self):
        result = CreateSLURPDistributionFromLogisticRegression(
            logistic_regression_model=self.logit_model,
            list_of_prediction_values=[0.5, -0.5],
            number_of_trials=200,
            plot_simulation_results=False,
            return_format='array'
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 200)

    def test_probabilities_are_within_valid_range(self):
        result = CreateSLURPDistributionFromLogisticRegression(
            logistic_regression_model=self.logit_model,
            list_of_prediction_values=[0.5, -0.5],
            number_of_trials=500,
            plot_simulation_results=False,
            return_format='array'
        )
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1).all())

    def test_reproducible_with_global_seed(self):
        # rmetalog draws from NumPy's legacy global random state, so reproducibility
        # currently depends on seeding that global state before calling the function.
        np.random.seed(123)
        result_1 = CreateSLURPDistributionFromLogisticRegression(
            logistic_regression_model=self.logit_model,
            list_of_prediction_values=[0.5, -0.5],
            number_of_trials=200,
            plot_simulation_results=False,
            return_format='array'
        )
        np.random.seed(123)
        result_2 = CreateSLURPDistributionFromLogisticRegression(
            logistic_regression_model=self.logit_model,
            list_of_prediction_values=[0.5, -0.5],
            number_of_trials=200,
            plot_simulation_results=False,
            return_format='array'
        )
        np.testing.assert_array_equal(result_1, result_2)

    def test_non_logistic_model_raises(self):
        with self.assertRaises(ValueError):
            CreateSLURPDistributionFromLogisticRegression(
                logistic_regression_model=self.ols_model,
                list_of_prediction_values=[0.5, -0.5],
                plot_simulation_results=False
            )

    def test_mismatched_prediction_values_length_raises(self):
        with self.assertRaises(ValueError):
            CreateSLURPDistributionFromLogisticRegression(
                logistic_regression_model=self.logit_model,
                list_of_prediction_values=[0.5],
                plot_simulation_results=False
            )

    def test_invalid_prediction_interval_raises(self):
        with self.assertRaises(ValueError):
            CreateSLURPDistributionFromLogisticRegression(
                logistic_regression_model=self.logit_model,
                list_of_prediction_values=[0.5, -0.5],
                prediction_interval=1.5,
                plot_simulation_results=False
            )
        with self.assertRaises(ValueError):
            CreateSLURPDistributionFromLogisticRegression(
                logistic_regression_model=self.logit_model,
                list_of_prediction_values=[0.5, -0.5],
                prediction_interval=0,
                plot_simulation_results=False
            )

    @patch('matplotlib.pyplot.show')
    def test_plot_is_shown(self, mock_show):
        CreateSLURPDistributionFromLogisticRegression(
            logistic_regression_model=self.logit_model,
            list_of_prediction_values=[0.5, -0.5],
            number_of_trials=200,
            plot_simulation_results=True,
            title_for_plot='Test Title',
            subtitle_for_plot='Test Subtitle'
        )
        self.assertTrue(mock_show.called)

    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()
