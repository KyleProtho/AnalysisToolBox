import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
from shapely.geometry import Polygon, mapping

from analysistoolbox.geospatial_analysis.CreateChoroplethMapFromNames import (
    CreateChoroplethMapFromNames,
    _GEOBOUNDARIES_CACHE,
)


def _square(x, y, size=1.0):
    """Build a small square Polygon anchored at (x, y), for fake boundary geometry."""
    return Polygon([(x, y), (x, y + size), (x + size, y + size), (x + size, y)])


def _geojson_feature(name, geometry, extra_properties=None):
    properties = {'shapeName': name}
    if extra_properties:
        properties.update(extra_properties)
    return {
        'type': 'Feature',
        'properties': properties,
        'geometry': mapping(geometry),
    }


def _mock_geoboundaries_response(iso3, admin_level, features):
    """Build the (metadata_response, geojson_response) pair returned for one API call."""
    download_url = f'https://example.com/{iso3}_{admin_level}.geojson'
    metadata_response = MagicMock()
    metadata_response.raise_for_status.return_value = None
    metadata_response.json.return_value = {'gjDownloadURL': download_url}

    geojson_response = MagicMock()
    geojson_response.raise_for_status.return_value = None
    geojson_response.json.return_value = {'type': 'FeatureCollection', 'features': features}

    return download_url, metadata_response, geojson_response


class TestCreateChoroplethMapFromNames(unittest.TestCase):

    def setUp(self):
        # Avoid cross-test contamination of the session-level geoBoundaries cache
        _GEOBOUNDARIES_CACHE.clear()

    def _patch_requests_get(self, metadata_url_map, geojson_url_map):
        """Return a requests.get side_effect that serves metadata and geojson responses
        keyed by URL, mimicking the two-hop geoBoundaries API + download flow."""
        def side_effect(url, headers=None, timeout=None):
            if url in metadata_url_map:
                return metadata_url_map[url]
            if url in geojson_url_map:
                return geojson_url_map[url]
            raise AssertionError(f"Unexpected URL requested: {url}")
        return side_effect

    # ------------------------------------------------------------------
    # Country-level matching via geoBoundaries (default boundary_source)
    # ------------------------------------------------------------------
    def test_country_level_matching_geoboundaries(self):
        kenya_feature = _geojson_feature('Kenya', _square(0, 0))
        download_url, metadata_response, geojson_response = _mock_geoboundaries_response(
            'KEN', 'ADM0', [kenya_feature]
        )
        metadata_url = 'https://www.geoboundaries.org/api/current/gbOpen/KEN/ADM0/'

        df = pd.DataFrame({
            'country': ['Kenya', 'Not A Real Country XYZ'],
            'incident_count': [12, 1],
        })

        with patch('requests.get') as mock_get:
            mock_get.side_effect = self._patch_requests_get(
                {metadata_url: metadata_response},
                {download_url: geojson_response},
            )
            matched, unmatched = CreateChoroplethMapFromNames(
                df,
                location_column='country',
                value_column='incident_count',
                geography_level='country',
                boundary_source='geoboundaries',
            )

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]['matched_boundary_name'], 'Kenya')
        self.assertEqual(matched.iloc[0]['match_confidence'], 100)
        self.assertIn('geometry', matched.columns)

        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched.iloc[0]['country'], 'Not A Real Country XYZ')
        self.assertIn('best_fuzzy_score', unmatched.columns)

    # ------------------------------------------------------------------
    # U.S. state matching via boundary_source='census'
    # ------------------------------------------------------------------
    def test_us_state_matching_census(self):
        states_gdf_data = {
            'NAME': ['California', 'Texas', 'New York'],
            'geometry': [_square(0, 0), _square(2, 0), _square(4, 0)],
        }
        import geopandas as gpd
        fake_states_gdf = gpd.GeoDataFrame(states_gdf_data, geometry='geometry', crs='EPSG:4326')

        df = pd.DataFrame({
            'state': ['California', 'Texas'],
            'sales': [500000, 420000],
        })

        with patch(
            'analysistoolbox.data_collection.FetchUSShapefile.FetchUSShapefile',
            return_value=fake_states_gdf,
        ) as mock_fetch:
            matched, unmatched = CreateChoroplethMapFromNames(
                df,
                location_column='state',
                value_column='sales',
                geography_level='us_state',
                boundary_source='census',
                census_year=2021,
            )

        mock_fetch.assert_called_once_with(state=None, geography='states', census_year=2021)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(unmatched), 0)
        self.assertSetEqual(set(matched['matched_boundary_name']), {'California', 'Texas'})
        self.assertTrue((matched['match_confidence'] == 100).all())

    # ------------------------------------------------------------------
    # U.S. state matching via boundary_source='geoboundaries' (admin1 + country_column)
    # ------------------------------------------------------------------
    def test_us_state_matching_geoboundaries_admin1(self):
        usa_feature = _geojson_feature('California', _square(0, 0))
        download_url, metadata_response, geojson_response = _mock_geoboundaries_response(
            'USA', 'ADM1', [usa_feature]
        )
        metadata_url = 'https://www.geoboundaries.org/api/current/gbOpen/USA/ADM1/'

        df = pd.DataFrame({
            'state': ['California'],
            'country': ['United States'],
            'sales': [500000],
        })

        with patch('requests.get') as mock_get:
            mock_get.side_effect = self._patch_requests_get(
                {metadata_url: metadata_response},
                {download_url: geojson_response},
            )
            matched, unmatched = CreateChoroplethMapFromNames(
                df,
                location_column='state',
                value_column='sales',
                geography_level='admin1',
                country_column='country',
                boundary_source='geoboundaries',
            )

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]['matched_boundary_name'], 'California')
        self.assertEqual(len(unmatched), 0)

    # ------------------------------------------------------------------
    # Ambiguous name resolved correctly via country_column
    # ------------------------------------------------------------------
    def test_ambiguous_name_resolved_via_country_column(self):
        # Two different countries both have an admin1 division named "Central"
        spain_feature = _geojson_feature('Central', _square(0, 0))
        argentina_feature = _geojson_feature('Central', _square(10, 10))

        esp_url, esp_meta, esp_geojson = _mock_geoboundaries_response('ESP', 'ADM1', [spain_feature])
        arg_url, arg_meta, arg_geojson = _mock_geoboundaries_response('ARG', 'ADM1', [argentina_feature])

        esp_metadata_url = 'https://www.geoboundaries.org/api/current/gbOpen/ESP/ADM1/'
        arg_metadata_url = 'https://www.geoboundaries.org/api/current/gbOpen/ARG/ADM1/'

        df = pd.DataFrame({
            'region': ['Central', 'Central'],
            'country': ['Spain', 'Argentina'],
            'value': [1, 2],
        })

        with patch('requests.get') as mock_get:
            mock_get.side_effect = self._patch_requests_get(
                {esp_metadata_url: esp_meta, arg_metadata_url: arg_meta},
                {esp_url: esp_geojson, arg_url: arg_geojson},
            )
            matched, unmatched = CreateChoroplethMapFromNames(
                df,
                location_column='region',
                value_column='value',
                geography_level='admin1',
                country_column='country',
                boundary_source='geoboundaries',
            )

        self.assertEqual(len(matched), 2)
        self.assertEqual(len(unmatched), 0)

        spain_row = matched[matched['country'] == 'Spain'].iloc[0]
        argentina_row = matched[matched['country'] == 'Argentina'].iloc[0]

        # Each row should be matched to the boundary from its own country, not the other's
        self.assertFalse(spain_row['geometry'].equals(argentina_row['geometry']))
        self.assertTrue(spain_row['geometry'].equals(_square(0, 0)))
        self.assertTrue(argentina_row['geometry'].equals(_square(10, 10)))

    # ------------------------------------------------------------------
    # Unmatchable name lands in unmatched_names_df instead of raising
    # ------------------------------------------------------------------
    def test_unmatchable_name_does_not_raise(self):
        kenya_feature = _geojson_feature('Kenya', _square(0, 0))
        download_url, metadata_response, geojson_response = _mock_geoboundaries_response(
            'KEN', 'ADM0', [kenya_feature]
        )
        metadata_url = 'https://www.geoboundaries.org/api/current/gbOpen/KEN/ADM0/'

        df = pd.DataFrame({
            'country': ['Kenya', '###totally_unmatchable_gibberish###'],
            'value': [1, 2],
        })

        with patch('requests.get') as mock_get:
            mock_get.side_effect = self._patch_requests_get(
                {metadata_url: metadata_response},
                {download_url: geojson_response},
            )
            try:
                matched, unmatched = CreateChoroplethMapFromNames(
                    df,
                    location_column='country',
                    value_column='value',
                    geography_level='country',
                    boundary_source='geoboundaries',
                )
            except Exception as e:
                self.fail(f"Unmatchable rows should not raise, but got: {e}")

        self.assertEqual(len(matched), 1)
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched.iloc[0]['country'], '###totally_unmatchable_gibberish###')

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def test_missing_location_column_raises(self):
        df = pd.DataFrame({'value': [1, 2]})
        with self.assertRaises(ValueError):
            CreateChoroplethMapFromNames(
                df, location_column='missing', value_column='value',
                geography_level='country',
            )

    def test_missing_value_column_raises(self):
        df = pd.DataFrame({'country': ['Kenya']})
        with self.assertRaises(ValueError):
            CreateChoroplethMapFromNames(
                df, location_column='country', value_column='missing',
                geography_level='country',
            )

    def test_ambiguous_auto_geography_level_raises(self):
        df = pd.DataFrame({
            'place': ['asdkfj', 'qweoiruq', 'zxcvbnm123'],
            'value': [1, 2, 3],
        })
        with self.assertRaises(ValueError):
            CreateChoroplethMapFromNames(df, location_column='place', value_column='value')


if __name__ == '__main__':
    unittest.main()
