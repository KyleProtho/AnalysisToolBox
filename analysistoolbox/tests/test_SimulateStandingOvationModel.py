# Load packages
import unittest
import random as pyrandom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from analysistoolbox.simulations import SimulateStandingOvationModel

class TestSimulateStandingOvationModel(unittest.TestCase):

    def test_returns_summary_dataframe(self):
        result = SimulateStandingOvationModel(
            qualities=(0.4, 0.7),
            hall_shape=(10, 12),
            plot_simulation_results=False
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertListEqual(
            list(result.columns),
            ['Quality', 'Converged', 'Steps', 'Initial Share Standing', 'Final Share Standing',
             'Share Stood On Own Judgment', 'Share Stood From Peer Pressure', 'Share Pioneers', 'Ovation']
        )

    def test_return_format_dict(self):
        result = SimulateStandingOvationModel(
            qualities=(0.55,),
            hall_shape=(8, 16),
            plot_simulation_results=False,
            return_format='dict'
        )
        self.assertIn('summary', result)
        self.assertIn('results', result)
        self.assertIsNone(result['sweep'])
        run = result['results'][0.55]
        self.assertEqual(run['frames'][0].shape, (8, 16))
        self.assertEqual(len(run['frames']), run['steps'] + 1)
        self.assertEqual(result['peer_thresholds'].shape, (8, 16))

    def test_breakdown_sums_to_final_share(self):
        result = SimulateStandingOvationModel(
            qualities=(0.45, 0.55, 0.65),
            hall_shape=(12, 12),
            number_of_pioneers=5,
            plot_simulation_results=False
        )
        np.testing.assert_allclose(
            result['Share Stood On Own Judgment'] + result['Share Stood From Peer Pressure'] + result['Share Pioneers'],
            result['Final Share Standing']
        )

    def test_no_peer_pressure_means_no_cascade(self):
        result = SimulateStandingOvationModel(
            qualities=(0.5, 0.6),
            peer_threshold=1.0,
            hall_shape=(10, 10),
            plot_simulation_results=False
        )
        np.testing.assert_allclose(result['Initial Share Standing'], result['Final Share Standing'])
        self.assertTrue((result['Share Stood From Peer Pressure'] == 0).all())
        self.assertTrue((result['Steps'] == 0).all())

    def test_peer_pressure_spreads_standing(self):
        result = SimulateStandingOvationModel(
            qualities=(0.55,),
            peer_threshold=0.3,
            hall_shape=(20, 30),
            plot_simulation_results=False
        )
        self.assertGreater(result['Final Share Standing'].iloc[0], result['Initial Share Standing'].iloc[0])
        self.assertTrue(result['Converged'].iloc[0])

    def test_no_noise_is_all_or_nothing(self):
        result = SimulateStandingOvationModel(
            qualities=(0.5, 0.7),
            quality_threshold=0.6,
            noise=0.0,
            hall_shape=(5, 5),
            plot_simulation_results=False
        )
        self.assertEqual(result['Final Share Standing'].iloc[0], 0)
        self.assertEqual(result['Final Share Standing'].iloc[1], 1)

    def test_front_row_cannot_be_pressured_with_cone(self):
        result = SimulateStandingOvationModel(
            qualities=(0.55,),
            peer_threshold=0.0,
            hall_shape=(10, 10),
            plot_simulation_results=False,
            return_format='dict'
        )
        frames = result['results'][0.55]['frames']
        np.testing.assert_array_equal(frames[0][0], frames[-1][0])

    def test_front_pioneers_beat_back_pioneers(self):
        kwargs = dict(qualities=(0.4,), number_of_pioneers=15, hall_shape=(20, 30),
                      plot_simulation_results=False)
        front = SimulateStandingOvationModel(pioneer_placement='front', **kwargs)
        back = SimulateStandingOvationModel(pioneer_placement='back', **kwargs)
        self.assertGreater(front['Final Share Standing'].iloc[0], back['Final Share Standing'].iloc[0])

    def test_pioneers_are_placed_correctly(self):
        result = SimulateStandingOvationModel(
            qualities=(0.0,),
            number_of_pioneers=4,
            pioneer_placement='back',
            hall_shape=(5, 6),
            plot_simulation_results=False,
            return_format='dict'
        )
        pioneers = result['pioneers']
        self.assertEqual(pioneers.sum(), 4)
        self.assertEqual(pioneers[-1].sum(), 4)
        # Center seats of the back row are filled first
        self.assertTrue(pioneers[-1, 1:5].all())

    def test_moore_lets_front_row_be_pressured(self):
        result = SimulateStandingOvationModel(
            qualities=(0.0,),
            peer_threshold=0.0,
            noise=0.0,
            number_of_pioneers=1,
            pioneer_placement='back',
            neighborhood='moore',
            hall_shape=(5, 5),
            plot_simulation_results=False
        )
        self.assertEqual(result['Final Share Standing'].iloc[0], 1)

    def test_can_sit_back_down(self):
        kwargs = dict(qualities=(0.6,), peer_threshold=0.5, hall_shape=(20, 30),
                      neighborhood='moore', plot_simulation_results=False, return_format='dict')
        stay = SimulateStandingOvationModel(can_sit_back_down=False, **kwargs)
        sit = SimulateStandingOvationModel(can_sit_back_down=True, **kwargs)
        stay_history = stay['results'][0.6]['share_history']
        sit_history = sit['results'][0.6]['share_history']
        # Without sitting back down, the share standing never decreases
        self.assertTrue(all(b >= a for a, b in zip(stay_history, stay_history[1:])))
        self.assertNotEqual(stay_history, sit_history)

    def test_all_qualities_share_the_same_audience(self):
        result = SimulateStandingOvationModel(
            qualities=(0.4, 0.7),
            hall_shape=(8, 8),
            plot_simulation_results=False,
            return_format='dict'
        )
        np.testing.assert_allclose(
            result['results'][0.7]['signals'] - result['results'][0.4]['signals'],
            0.3
        )

    def test_quality_sweep(self):
        result = SimulateStandingOvationModel(
            qualities=(0.5,),
            hall_shape=(8, 8),
            include_quality_sweep=True,
            sweep_qualities=(0.4, 0.6, 0.8),
            sweep_number_of_seeds=2,
            plot_simulation_results=False,
            return_format='dict'
        )
        self.assertIsInstance(result['sweep'], pd.DataFrame)
        self.assertEqual(len(result['sweep']), 6)

    def test_reproducible_with_seed(self):
        kwargs = dict(qualities=(0.55,), hall_shape=(10, 10), peer_spread=0.2,
                      plot_simulation_results=False, return_format='dict')
        result_1 = SimulateStandingOvationModel(random_seed=412, **kwargs)
        result_2 = SimulateStandingOvationModel(random_seed=412, **kwargs)
        np.testing.assert_array_equal(result_1['results'][0.55]['frames'][-1], result_2['results'][0.55]['frames'][-1])

    def test_different_seed_gives_different_results(self):
        kwargs = dict(qualities=(0.55,), hall_shape=(10, 10),
                      plot_simulation_results=False, return_format='dict')
        result_1 = SimulateStandingOvationModel(random_seed=412, **kwargs)
        result_2 = SimulateStandingOvationModel(random_seed=413, **kwargs)
        self.assertFalse(np.array_equal(result_1['results'][0.55]['signals'], result_2['results'][0.55]['signals']))

    def test_does_not_mutate_global_random_state(self):
        pyrandom.seed(999)
        py_state_before = pyrandom.getstate()
        np.random.seed(999)
        np_state_before = np.random.get_state()

        SimulateStandingOvationModel(
            qualities=(0.55,),
            hall_shape=(10, 10),
            number_of_pioneers=5,
            pioneer_placement='random',
            plot_simulation_results=False
        )

        self.assertEqual(pyrandom.getstate(), py_state_before)
        np.testing.assert_equal(np.random.get_state(), np_state_before)

    @patch('builtins.print')
    def test_print_step_by_step(self, mock_print):
        SimulateStandingOvationModel(
            qualities=(0.55,),
            hall_shape=(4, 6),
            print_step_by_step=True,
            plot_simulation_results=False
        )
        self.assertTrue(mock_print.called)

    def test_invalid_peer_threshold_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(peer_threshold=1.5, plot_simulation_results=False)

    def test_negative_noise_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(noise=-0.1, plot_simulation_results=False)

    def test_invalid_hall_shape_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(hall_shape=(10,), plot_simulation_results=False)

    def test_invalid_neighborhood_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(neighborhood='hexagonal', plot_simulation_results=False)

    def test_too_many_pioneers_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(hall_shape=(3, 3), number_of_pioneers=10, plot_simulation_results=False)

    def test_invalid_pioneer_placement_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(number_of_pioneers=2, pioneer_placement='aisle', plot_simulation_results=False)

    def test_duplicate_qualities_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(qualities=(0.5, 0.5), plot_simulation_results=False)

    def test_invalid_return_format_raises(self):
        with self.assertRaises(ValueError):
            SimulateStandingOvationModel(return_format='invalid', plot_simulation_results=False)

    @patch('matplotlib.pyplot.show')
    def test_plot_is_shown(self, mock_show):
        SimulateStandingOvationModel(
            qualities=(0.45, 0.6),
            hall_shape=(8, 12),
            number_of_pioneers=3,
            include_quality_sweep=True,
            sweep_qualities=(0.4, 0.8),
            sweep_number_of_seeds=1,
            caption_for_plot="Test caption",
            data_source_for_plot="Test source"
        )
        self.assertTrue(mock_show.called)

    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()
