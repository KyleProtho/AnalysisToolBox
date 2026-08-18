import unittest
import unittest.mock

import numpy as np
import pandas as pd

from analysistoolbox.geospatial_analysis import FindNearestPointOfInterest


class TestFindNearestPointOfInterest(unittest.TestCase):

    def setUp(self):
        # Three observations near Los Angeles
        self.observations_df = pd.DataFrame({
            'sighting_id': ['S1', 'S2', 'S3'],
            'lat': [34.0522, 34.1478, 33.9425],
            'lon': [-118.2437, -118.1445, -118.4081],
        })
        # Two reference facilities: one close to LA/S1, one close to S3
        self.poi_df = pd.DataFrame({
            'facility_name': ['Northgate Warehouse', 'Southside Depot'],
            'lat': [34.0600, 33.9500],
            'lon': [-118.2500, -118.4000],
        })

    def test_returns_one_row_per_observation_by_default(self):
        result = FindNearestPointOfInterest(
            self.observations_df, self.poi_df,
            id_column='sighting_id', poi_id_column='facility_name'
        )
        self.assertEqual(len(result), 3)
        self.assertIn('distance_km', result.columns)
        self.assertIn('nearest_poi_id', result.columns)

    def test_nearest_match_is_correct(self):
        result = FindNearestPointOfInterest(
            self.observations_df, self.poi_df,
            id_column='sighting_id', poi_id_column='facility_name'
        )
        s1_match = result[result['point_id'] == 'S1'].iloc[0]
        s3_match = result[result['point_id'] == 'S3'].iloc[0]
        self.assertEqual(s1_match['nearest_poi_id'], 'Northgate Warehouse')
        self.assertEqual(s3_match['nearest_poi_id'], 'Southside Depot')

    def test_number_of_neighbors_expands_rows(self):
        result = FindNearestPointOfInterest(
            self.observations_df, self.poi_df,
            id_column='sighting_id', poi_id_column='facility_name',
            number_of_neighbors=2
        )
        self.assertEqual(len(result), 6)
        self.assertListEqual(sorted(result['neighbor_rank'].unique().tolist()), [1, 2])

    def test_neighbor_rank_orders_by_distance(self):
        result = FindNearestPointOfInterest(
            self.observations_df, self.poi_df,
            id_column='sighting_id', poi_id_column='facility_name',
            number_of_neighbors=2
        )
        s1_matches = result[result['point_id'] == 'S1'].sort_values('neighbor_rank')
        self.assertTrue(
            s1_matches.iloc[0]['distance_km'] <= s1_matches.iloc[1]['distance_km']
        )

    def test_distance_unit_conversion(self):
        km_result = FindNearestPointOfInterest(
            self.observations_df, self.poi_df,
            id_column='sighting_id', poi_id_column='facility_name', distance_unit='km'
        )
        mi_result = FindNearestPointOfInterest(
            self.observations_df, self.poi_df,
            id_column='sighting_id', poi_id_column='facility_name', distance_unit='mi'
        )
        self.assertAlmostEqual(
            mi_result.iloc[0]['distance_mi'] / km_result.iloc[0]['distance_km'],
            0.621,
            delta=0.01,
        )

    def test_uses_index_when_no_id_columns_given(self):
        result = FindNearestPointOfInterest(self.observations_df, self.poi_df)
        self.assertIn('point_id', result.columns)
        self.assertIn('nearest_poi_id', result.columns)

    def test_missing_coordinates_are_dropped(self):
        df = self.observations_df.copy()
        df.loc[0, 'lat'] = np.nan
        with unittest.mock.patch('builtins.print') as mock_print:
            result = FindNearestPointOfInterest(
                df, self.poi_df, id_column='sighting_id', poi_id_column='facility_name'
            )
            printed_messages = ' '.join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertEqual(len(result), 2)
        self.assertIn('missing coordinates', printed_messages.lower())

    def test_missing_longitude_column_raises(self):
        with self.assertRaises(ValueError):
            FindNearestPointOfInterest(self.observations_df, self.poi_df, longitude_column='missing')

    def test_missing_poi_latitude_column_raises(self):
        with self.assertRaises(ValueError):
            FindNearestPointOfInterest(self.observations_df, self.poi_df, poi_latitude_column='missing')

    def test_invalid_distance_unit_raises(self):
        with self.assertRaises(ValueError):
            FindNearestPointOfInterest(self.observations_df, self.poi_df, distance_unit='furlongs')

    def test_number_of_neighbors_below_one_raises(self):
        with self.assertRaises(ValueError):
            FindNearestPointOfInterest(self.observations_df, self.poi_df, number_of_neighbors=0)

    def test_number_of_neighbors_exceeding_poi_count_raises(self):
        with self.assertRaises(ValueError):
            FindNearestPointOfInterest(self.observations_df, self.poi_df, number_of_neighbors=5)

    def test_all_missing_observation_coordinates_raises(self):
        df = self.observations_df.copy()
        df['lat'] = np.nan
        with self.assertRaises(ValueError):
            FindNearestPointOfInterest(df, self.poi_df)


if __name__ == '__main__':
    unittest.main()
