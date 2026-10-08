# Load packages
import unittest
import random as pyrandom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from analysistoolbox.simulations import CreateCorrelatedSIPs

class TestCreateCorrelatedSIPs(unittest.TestCase):

    def test_reproducible_with_seed(self):
        result_1 = CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=500,
            random_seed=412,
            plot_simulation_results=False
        )
        result_2 = CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=500,
            random_seed=412,
            plot_simulation_results=False
        )
        pd.testing.assert_frame_equal(result_1, result_2)

    def test_different_seed_gives_different_results(self):
        result_1 = CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=500,
            random_seed=412,
            plot_simulation_results=False
        )
        result_2 = CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=500,
            random_seed=413,
            plot_simulation_results=False
        )
        self.assertFalse(result_1.equals(result_2))

    def test_does_not_mutate_global_random_state(self):
        pyrandom.seed(999)
        py_state_before = pyrandom.getstate()
        np.random.seed(999)
        np_state_before = np.random.get_state()

        CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=200,
            random_seed=412,
            plot_simulation_results=False
        )

        self.assertEqual(pyrandom.getstate(), py_state_before)
        np.testing.assert_equal(np.random.get_state(), np_state_before)

    def test_returns_dataframe_with_expected_columns(self):
        result = CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=100,
            variable_1_name='Patient Age',
            variable_2_name='Recovery Days',
            plot_simulation_results=False
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 100)
        self.assertEqual(result.columns.tolist(), ['Patient Age', 'Recovery Days'])

    def test_achieves_target_correlation(self):
        target_correlation = 0.65
        result = CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=target_correlation,
            number_of_trials=10000,
            random_seed=412,
            plot_simulation_results=False
        )
        observed_correlation = result['Variable 1'].corr(result['Variable 2'])
        self.assertAlmostEqual(observed_correlation, target_correlation, delta=0.02)

    def test_matches_requested_means_and_standard_deviations(self):
        result = CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=10000,
            random_seed=412,
            plot_simulation_results=False
        )
        self.assertAlmostEqual(result['Variable 1'].mean(), 65.0, delta=0.5)
        self.assertAlmostEqual(result['Variable 1'].std(), 12.0, delta=0.5)
        self.assertAlmostEqual(result['Variable 2'].mean(), 14.0, delta=0.5)
        self.assertAlmostEqual(result['Variable 2'].std(), 4.0, delta=0.5)

    @patch('matplotlib.pyplot.show')
    def test_plot_is_shown(self, mock_show):
        CreateCorrelatedSIPs(
            mean_of_variable_1=65.0,
            std_of_variable_1=12.0,
            mean_of_variable_2=14.0,
            std_of_variable_2=4.0,
            correlation=0.65,
            number_of_trials=100,
            plot_simulation_results=True
        )
        self.assertTrue(mock_show.called)

    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()
