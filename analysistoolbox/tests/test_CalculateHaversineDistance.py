import unittest
import unittest.mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysistoolbox.geospatial_analysis import CalculateHaversineDistance


class TestCalculateHaversineDistance(unittest.TestCase):

    def setUp(self):
        # Los Angeles, San Diego, and a duplicate of Los Angeles
        self.points_df = pd.DataFrame({
            'site_name': ['Los Angeles', 'San Diego', 'Los Angeles Dup'],
            'lat': [34.0522, 32.7157, 34.0522],
            'lon': [-118.2437, -117.1611, -118.2437],
        })

    def tearDown(self):
        plt.clf()

    def test_long_format_has_one_row_per_unique_pair(self):
        result = CalculateHaversineDistance(self.points_df, id_column='site_name')
        # Duplicate 'Los Angeles Dup' point is dropped, leaving 2 unique points -> 1 pair
        self.assertEqual(len(result), 1)
        self.assertIn('distance_km', result.columns)

    def test_known_distance_between_la_and_san_diego(self):
        result = CalculateHaversineDistance(self.points_df, id_column='site_name')
        # LA <-> San Diego great-circle distance is well documented at ~179 km
        self.assertAlmostEqual(result.iloc[0]['distance_km'], 179, delta=5)

    def test_duplicate_points_are_dropped(self):
        with unittest.mock.patch('builtins.print') as mock_print:
            CalculateHaversineDistance(self.points_df, id_column='site_name')
            printed_messages = ' '.join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn('duplicate', printed_messages.lower())

    def test_matrix_format_is_square_and_symmetric(self):
        result = CalculateHaversineDistance(
            self.points_df, id_column='site_name', output_format='matrix'
        )
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.allclose(result.values, result.values.T))
        self.assertTrue(np.allclose(np.diag(result.values), 0))

    def test_distance_unit_conversion(self):
        km_result = CalculateHaversineDistance(
            self.points_df, id_column='site_name', distance_unit='km'
        )
        mi_result = CalculateHaversineDistance(
            self.points_df, id_column='site_name', distance_unit='mi'
        )
        # Miles should be roughly 0.621 times the kilometer distance
        self.assertAlmostEqual(
            mi_result.iloc[0]['distance_mi'] / km_result.iloc[0]['distance_km'],
            0.621,
            delta=0.01,
        )

    def test_uses_index_when_no_id_column_given(self):
        result = CalculateHaversineDistance(self.points_df)
        self.assertIn('point_1', result.columns)
        self.assertIn('point_2', result.columns)

    def test_missing_longitude_column_raises(self):
        with self.assertRaises(ValueError):
            CalculateHaversineDistance(self.points_df, longitude_column='missing')

    def test_missing_id_column_raises(self):
        with self.assertRaises(ValueError):
            CalculateHaversineDistance(self.points_df, id_column='missing')

    def test_invalid_distance_unit_raises(self):
        with self.assertRaises(ValueError):
            CalculateHaversineDistance(self.points_df, distance_unit='furlongs')

    def test_invalid_output_format_raises(self):
        with self.assertRaises(ValueError):
            CalculateHaversineDistance(self.points_df, output_format='wide')

    def test_too_few_unique_points_raises(self):
        df = pd.DataFrame({'lat': [34.0522, 34.0522], 'lon': [-118.2437, -118.2437]})
        with self.assertRaises(ValueError):
            CalculateHaversineDistance(df)

    def test_plot_connections_runs_without_error(self):
        # Should not raise when generating the connecting-line visualization
        CalculateHaversineDistance(self.points_df, id_column='site_name', plot_connections=True)


if __name__ == '__main__':
    unittest.main()
