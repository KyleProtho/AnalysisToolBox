# Load packages
import unittest
import random as pyrandom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from analysistoolbox.simulations import SimulateSchellingSegregation

class TestSimulateSchellingSegregation(unittest.TestCase):

    def test_returns_summary_dataframe(self):
        result = SimulateSchellingSegregation(
            tolerances=(0.25, 0.75),
            grid_shape=(12, 12),
            max_steps=50,
            plot_simulation_results=False
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertListEqual(
            list(result.columns),
            ['Tolerance', 'Similarity Threshold', 'Converged', 'Steps',
             'Initial Similarity', 'Final Similarity', 'Final Unhappy Agents']
        )
        np.testing.assert_allclose(result['Similarity Threshold'], [0.75, 0.25])

    def test_return_format_dict(self):
        result = SimulateSchellingSegregation(
            tolerances=(0.5,),
            grid_shape=(10, 10),
            max_steps=50,
            plot_simulation_results=False,
            return_format='dict'
        )
        self.assertIn('summary', result)
        self.assertIn('results', result)
        self.assertIsNone(result['sweep'])
        run = result['results'][0.5]
        self.assertEqual(run['initial'].shape, (10, 10))
        self.assertEqual(run['final'].shape, (10, 10))

    def test_full_tolerance_never_moves(self):
        result = SimulateSchellingSegregation(
            tolerances=(1.0,),
            grid_shape=(10, 10),
            plot_simulation_results=False,
            return_format='dict'
        )
        run = result['results'][1.0]
        np.testing.assert_array_equal(run['initial'], run['final'])
        self.assertTrue(run['converged'])
        self.assertEqual(run['steps'], 0)

    def test_moderate_tolerance_converges_and_segregates(self):
        result = SimulateSchellingSegregation(
            tolerances=(0.5,),
            grid_shape=(20, 20),
            max_steps=300,
            plot_simulation_results=False
        )
        self.assertTrue(result['Converged'].iloc[0])
        self.assertEqual(result['Final Unhappy Agents'].iloc[0], 0)
        self.assertGreater(result['Final Similarity'].iloc[0], result['Initial Similarity'].iloc[0])

    def test_agent_counts_are_preserved(self):
        result = SimulateSchellingSegregation(
            tolerances=(0.3,),
            grid_shape=(20, 20),
            number_of_factions=3,
            faction_fractions=(0.5, 0.3, 0.2),
            max_steps=20,
            plot_simulation_results=False,
            return_format='dict'
        )
        run = result['results'][0.3]
        for value in range(4):
            self.assertEqual((run['initial'] == value).sum(), (run['final'] == value).sum())
        # 400 cells, 40 empty, 360 occupied split 50/30/20
        self.assertEqual((run['initial'] == 0).sum(), 40)
        self.assertEqual((run['initial'] == 1).sum(), 180)
        self.assertEqual((run['initial'] == 2).sum(), 108)
        self.assertEqual((run['initial'] == 3).sum(), 72)

    def test_all_tolerances_share_the_same_starting_grid(self):
        result = SimulateSchellingSegregation(
            tolerances=(0.2, 0.6),
            grid_shape=(10, 10),
            max_steps=10,
            plot_simulation_results=False,
            return_format='dict'
        )
        np.testing.assert_array_equal(result['results'][0.2]['initial'], result['results'][0.6]['initial'])

    def test_one_dimensional_starting_grid(self):
        row = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0]
        result = SimulateSchellingSegregation(
            tolerances=(0.5,),
            starting_grid=row,
            max_steps=12,
            plot_simulation_results=False,
            return_format='dict'
        )
        run = result['results'][0.5]
        np.testing.assert_array_equal(run['initial'], row)
        self.assertEqual(run['final'].shape, (12,))
        # Alternating row: every agent has only unlike neighbors at the start
        self.assertEqual(result['summary']['Initial Similarity'].iloc[0], 0)

    def test_neighborhood_changes_similarity(self):
        # Checkerboard: Moore neighbors include like diagonals, von Neumann neighbors are all unlike
        checkerboard = (np.indices((6, 6)).sum(axis=0) % 2) + 1
        moore = SimulateSchellingSegregation(
            tolerances=(1.0,),
            starting_grid=checkerboard,
            neighborhood='moore',
            plot_simulation_results=False
        )
        von_neumann = SimulateSchellingSegregation(
            tolerances=(1.0,),
            starting_grid=checkerboard,
            neighborhood='von_neumann',
            plot_simulation_results=False
        )
        self.assertGreater(moore['Initial Similarity'].iloc[0], 0)
        self.assertEqual(von_neumann['Initial Similarity'].iloc[0], 0)

    def test_wrap_edges_changes_corner_neighbors(self):
        # Each faction fills half the grid; wrapping makes edge cells see the other half
        halves = np.ones((6, 6), dtype=int)
        halves[:, 3:] = 2
        no_wrap = SimulateSchellingSegregation(
            tolerances=(1.0,),
            starting_grid=halves,
            wrap_edges=False,
            plot_simulation_results=False
        )
        wrap = SimulateSchellingSegregation(
            tolerances=(1.0,),
            starting_grid=halves,
            wrap_edges=True,
            plot_simulation_results=False
        )
        self.assertGreater(no_wrap['Initial Similarity'].iloc[0], wrap['Initial Similarity'].iloc[0])

    def test_tolerance_sweep(self):
        result = SimulateSchellingSegregation(
            tolerances=(0.5,),
            grid_shape=(10, 10),
            max_steps=20,
            include_tolerance_sweep=True,
            sweep_tolerances=(0.0, 0.5, 1.0),
            sweep_number_of_seeds=2,
            plot_simulation_results=False,
            return_format='dict'
        )
        self.assertIsInstance(result['sweep'], pd.DataFrame)
        self.assertEqual(len(result['sweep']), 6)

    def test_reproducible_with_seed(self):
        kwargs = dict(tolerances=(0.3,), grid_shape=(10, 10), max_steps=20,
                      plot_simulation_results=False, return_format='dict')
        result_1 = SimulateSchellingSegregation(random_seed=412, **kwargs)
        result_2 = SimulateSchellingSegregation(random_seed=412, **kwargs)
        np.testing.assert_array_equal(result_1['results'][0.3]['final'], result_2['results'][0.3]['final'])

    def test_different_seed_gives_different_results(self):
        kwargs = dict(tolerances=(0.3,), grid_shape=(10, 10), max_steps=20,
                      plot_simulation_results=False, return_format='dict')
        result_1 = SimulateSchellingSegregation(random_seed=412, **kwargs)
        result_2 = SimulateSchellingSegregation(random_seed=413, **kwargs)
        self.assertFalse(np.array_equal(result_1['results'][0.3]['initial'], result_2['results'][0.3]['initial']))

    def test_does_not_mutate_global_random_state(self):
        pyrandom.seed(999)
        py_state_before = pyrandom.getstate()
        np.random.seed(999)
        np_state_before = np.random.get_state()

        SimulateSchellingSegregation(
            tolerances=(0.3,),
            grid_shape=(10, 10),
            max_steps=10,
            plot_simulation_results=False
        )

        self.assertEqual(pyrandom.getstate(), py_state_before)
        np.testing.assert_equal(np.random.get_state(), np_state_before)

    def test_too_many_factions_raises(self):
        # 50% of the smaller dimension (8) is 4 factions
        SimulateSchellingSegregation(
            tolerances=(0.5,),
            grid_shape=(8, 20),
            number_of_factions=4,
            max_steps=5,
            plot_simulation_results=False
        )
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(
                tolerances=(0.5,),
                grid_shape=(8, 20),
                number_of_factions=5,
                plot_simulation_results=False
            )

    def test_fewer_than_two_factions_raises(self):
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(number_of_factions=1, plot_simulation_results=False)

    def test_invalid_faction_fractions_raises(self):
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(faction_fractions=(0.5, 0.3), number_of_factions=3,
                                         plot_simulation_results=False)
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(faction_fractions=(0.6, 0.6), plot_simulation_results=False)

    def test_starting_grid_with_unknown_faction_raises(self):
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(starting_grid=[1, 2, 3, 0, 1, 2], plot_simulation_results=False)

    def test_invalid_tolerance_raises(self):
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(tolerances=(0.5, 1.5), plot_simulation_results=False)

    def test_invalid_neighborhood_raises(self):
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(neighborhood='hexagonal', plot_simulation_results=False)

    def test_wrap_edges_on_small_grid_raises(self):
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(starting_grid=[[1, 2], [2, 1]], wrap_edges=True,
                                         plot_simulation_results=False)

    def test_invalid_return_format_raises(self):
        with self.assertRaises(ValueError):
            SimulateSchellingSegregation(return_format='invalid', plot_simulation_results=False)

    @patch('matplotlib.pyplot.show')
    def test_plot_is_shown(self, mock_show):
        SimulateSchellingSegregation(
            tolerances=(0.25, 0.75),
            grid_shape=(10, 10),
            number_of_factions=3,
            max_steps=10,
            include_tolerance_sweep=True,
            sweep_tolerances=(0.0, 1.0),
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
