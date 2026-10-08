# Load packages
import unittest
import random as pyrandom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from analysistoolbox.simulations import SimulateTimeBetweenEvents

class TestSimulateTimeBetweenEvents(unittest.TestCase):

    def test_reproducible_with_seed(self):
        result_1 = SimulateTimeBetweenEvents(
            expected_time_between_events=7.5,
            number_of_trials=200,
            random_seed=412,
            plot_simulation_results=False,
            return_format='array'
        )
        result_2 = SimulateTimeBetweenEvents(
            expected_time_between_events=7.5,
            number_of_trials=200,
            random_seed=412,
            plot_simulation_results=False,
            return_format='array'
        )
        np.testing.assert_array_equal(result_1, result_2)

    def test_different_seed_gives_different_results(self):
        result_1 = SimulateTimeBetweenEvents(
            expected_time_between_events=7.5,
            number_of_trials=200,
            random_seed=412,
            plot_simulation_results=False,
            return_format='array'
        )
        result_2 = SimulateTimeBetweenEvents(
            expected_time_between_events=7.5,
            number_of_trials=200,
            random_seed=413,
            plot_simulation_results=False,
            return_format='array'
        )
        self.assertFalse(np.array_equal(result_1, result_2))

    def test_does_not_mutate_global_random_state(self):
        pyrandom.seed(999)
        py_state_before = pyrandom.getstate()
        np.random.seed(999)
        np_state_before = np.random.get_state()

        SimulateTimeBetweenEvents(
            expected_time_between_events=7.5,
            number_of_trials=50,
            random_seed=412,
            plot_simulation_results=False,
            return_format='array'
        )

        self.assertEqual(pyrandom.getstate(), py_state_before)
        np.testing.assert_equal(np.random.get_state(), np_state_before)

    def test_return_format_dataframe(self):
        result = SimulateTimeBetweenEvents(
            expected_time_between_events=7.5,
            number_of_trials=50,
            plot_simulation_results=False,
            return_format='dataframe'
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 50)

    def test_invalid_return_format_raises(self):
        with self.assertRaises(ValueError):
            SimulateTimeBetweenEvents(
                expected_time_between_events=7.5,
                plot_simulation_results=False,
                return_format='invalid'
            )

    def test_non_positive_expected_time_raises(self):
        with self.assertRaises(ValueError):
            SimulateTimeBetweenEvents(
                expected_time_between_events=0,
                plot_simulation_results=False
            )

    @patch('matplotlib.pyplot.show')
    def test_plot_is_shown(self, mock_show):
        SimulateTimeBetweenEvents(
            expected_time_between_events=7.5,
            number_of_trials=50,
            plot_simulation_results=True
        )
        self.assertTrue(mock_show.called)

    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()
