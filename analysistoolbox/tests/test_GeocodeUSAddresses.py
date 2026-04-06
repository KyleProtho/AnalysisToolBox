import unittest
from unittest.mock import MagicMock, patch
import sys
import pandas as pd
import numpy as np

# Mock censusgeocode before it's imported locally in the function
# This is necessary because censusgeocode may not be installed in the environment
mock_cg = MagicMock()
sys.modules['censusgeocode'] = mock_cg

from analysistoolbox.data_processing.GeocodeUSAddresses import GeocodeUSAddresses

class TestGeocodeUSAddresses(unittest.TestCase):

    def setUp(self):
        # Reset the mock before each test to ensure a clean slate
        mock_cg.reset_mock()
        mock_cg.onelineaddress.side_effect = None
        mock_cg.onelineaddress.return_value = None

    def test_geocode_success(self):
        """Test successful geocoding of a valid address."""
        # Mock successful response from the U.S. Census Bureau service
        # Note: censusgeocode returns coordinates as {'x': longitude, 'y': latitude}
        mock_cg.onelineaddress.return_value = [
            {
                'coordinates': {'x': -122.084057, 'y': 37.422388}
            }
        ]
        
        df = pd.DataFrame({
            'address': ['1600 Amphitheatre Pkwy, Mountain View, CA 94043']
        })
        
        result = GeocodeUSAddresses(df, 'address')
        
        # Verify columns were added
        self.assertIn('Latitude', result.columns)
        self.assertIn('Longitude', result.columns)
        
        # Verify correct coordinates were mapped (x=longitude, y=latitude)
        self.assertEqual(result.iloc[0]['Latitude'], 37.422388)
        self.assertEqual(result.iloc[0]['Longitude'], -122.084057)
        
        # Verify the mock was called correctly
        mock_cg.onelineaddress.assert_called_once_with(
            '1600 Amphitheatre Pkwy, Mountain View, CA 94043', 
            returntype='locations'
        )

    def test_geocode_multiple_addresses(self):
        """Test geocoding multiple addresses in a single DataFrame."""
        def side_effect(address, returntype='locations'):
            if "Amphitheatre" in address:
                return [{'coordinates': {'x': -122.084, 'y': 37.422}}]
            elif "Apple Park" in address:
                return [{'coordinates': {'x': -122.009, 'y': 37.334}}]
            return []

        mock_cg.onelineaddress.side_effect = side_effect
        
        df = pd.DataFrame({
            'address': [
                '1600 Amphitheatre Pkwy, Mountain View, CA 94043',
                'One Apple Park Way, Cupertino, CA 95014',
                'Invalid Address'
            ]
        })
        
        result = GeocodeUSAddresses(df, 'address')
        
        self.assertEqual(result.iloc[0]['Latitude'], 37.422)
        self.assertEqual(result.iloc[1]['Latitude'], 37.334)
        self.assertTrue(np.isnan(result.iloc[2]['Latitude']))

    def test_geocode_no_match(self):
        """Test handling of addresses that return no matches from the API."""
        # Mock empty response
        mock_cg.onelineaddress.return_value = []
        
        df = pd.DataFrame({
            'address': ['This address definitely does not exist']
        })
        
        result = GeocodeUSAddresses(df, 'address')
        
        # Unmatched addresses should result in NaN coordinates
        self.assertTrue(np.isnan(result.iloc[0]['Latitude']))
        self.assertTrue(np.isnan(result.iloc[0]['Longitude']))

    def test_geocode_exception_handling(self):
        """Test that the function gracefully handles API or network exceptions."""
        # Mock an exception during the API call
        mock_cg.onelineaddress.side_effect = Exception("Service Unavailable")
        
        df = pd.DataFrame({
            'address': ['Connection Error Address']
        })
        
        # The function should handle exceptions and return NaN coordinates
        result = GeocodeUSAddresses(df, 'address')
        
        self.assertTrue(np.isnan(result.iloc[0]['Latitude']))
        self.assertTrue(np.isnan(result.iloc[0]['Longitude']))

    def test_custom_column_names(self):
        """Test geocoding with user-defined output column names for latitude and longitude."""
        mock_cg.onelineaddress.return_value = [
            {
                'coordinates': {'x': -74.0060, 'y': 40.7128}
            }
        ]
        
        df = pd.DataFrame({
            'full_str': ['New York, NY']
        })
        
        result = GeocodeUSAddresses(
            df, 
            'full_str', 
            latitude_column_name='lat', 
            longitude_column_name='lon'
        )
        
        # Verify custom column names were used
        self.assertIn('lat', result.columns)
        self.assertIn('lon', result.columns)
        self.assertEqual(result.iloc[0]['lat'], 40.7128)
        self.assertEqual(result.iloc[0]['lon'], -74.0060)

    def test_geocode_famous_buildings(self):
        """Test geocoding a dataframe of 5 famous U.S. buildings."""
        buildings_data = pd.DataFrame({
            'building': [
                'Empire State Building',
                'White House',
                'Space Needle',
                'Willis Tower',
                'Golden Gate Bridge Welcome Center'
            ],
            'address': [
                '350 5th Ave, New York, NY 10118',
                '1600 Pennsylvania Avenue NW, Washington, DC 20500',
                '400 Broad St, Seattle, WA 98109',
                '233 S Wacker Dr, Chicago, IL  Chicago, IL 60606',
                'Golden Gate Bridge, San Francisco, CA 94129'
            ]
        })

        # Define mock coordinates for each building (Longitude, Latitude)
        mock_coords = {
            '350 5th Ave, New York, NY 10118': (-73.9857, 40.7484),
            '1600 Pennsylvania Avenue NW, Washington, DC 20500': (-77.0365, 38.8977),
            '400 Broad St, Seattle, WA 98109': (-122.3493, 47.6205),
            '233 S Wacker Dr, Chicago, IL  Chicago, IL 60606': (-87.6359, 41.8789),
            'Golden Gate Bridge, San Francisco, CA 94129': (-122.4783, 37.8199)
        }

        def side_effect(address, returntype='locations'):
            if address in mock_coords:
                lon, lat = mock_coords[address]
                return [{'coordinates': {'x': lon, 'y': lat}}]
            return []

        mock_cg.onelineaddress.side_effect = side_effect

        result = GeocodeUSAddresses(buildings_data, 'address')

        # Verify all buildings were geocoded
        self.assertEqual(len(result), 5)
        self.assertFalse(result['Latitude'].isnull().any())
        self.assertFalse(result['Longitude'].isnull().any())

        # Verify specific coordinates
        self.assertAlmostEqual(result.iloc[0]['Latitude'], 40.7484, places=4)
        self.assertAlmostEqual(result.iloc[1]['Latitude'], 38.8977, places=4)
        self.assertAlmostEqual(result.iloc[2]['Latitude'], 47.6205, places=4)
        self.assertAlmostEqual(result.iloc[3]['Latitude'], 41.8789, places=4)
        self.assertAlmostEqual(result.iloc[4]['Latitude'], 37.8199, places=4)

if __name__ == '__main__':
    unittest.main()
