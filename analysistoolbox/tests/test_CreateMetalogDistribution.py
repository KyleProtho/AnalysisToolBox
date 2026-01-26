# Load packages
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from analysistoolbox.simulations import CreateMetalogDistribution

class TestCreateMetalogDistribution(unittest.TestCase):

    def setUp(self):
        # Data with integers
        self.int_df = pd.DataFrame({
            'vals': [10, 12, 12, 13, 14, 15, 18, 20, 22, 25, 30, 35, 40]
        })
        
        # Data with long decimal numbers
        self.decimal_df = pd.DataFrame({
            'vals': [0.123456789, 0.987654321, 0.456789012, 1.567890123, 0.678901234, 
                     0.789012345, 0.890123456, 0.901234567, 0.012345678, 0.123456789,
                     0.0000011145121344564981]
        })

    @patch('matplotlib.pyplot.show')
    def test_integers_basic(self, mock_show):
        """Test Metalog distribution with basic integer data."""
        result = CreateMetalogDistribution(
            self.int_df, 
            'vals', 
            number_of_samples=100, 
            show_summary=False,
            plot_metalog_distribution=True
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 100)
        self.assertTrue(mock_show.called)

    @patch('matplotlib.pyplot.show')
    def test_integers_bounded(self, mock_show):
        """Test Metalog distribution with bounded integer data."""
        # Use a small number of samples for speed
        result = CreateMetalogDistribution(
            self.int_df, 
            'vals', 
            lower_bound=0, 
            upper_bound=100,
            number_of_samples=50,
            show_summary=False
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 50)
        # Check that samples are within bounds (approximately, as metalog can sometimes slightly vary at tails depending on fit)
        # But logically they should respect bounds if the fit is good.
        self.assertTrue((result['vals'] >= 0).all())
        self.assertTrue((result['vals'] <= 100).all())

    @patch('matplotlib.pyplot.show')
    def test_decimals_long(self, mock_show):
        """Test Metalog distribution with long decimal numbers."""
        result = CreateMetalogDistribution(
            self.decimal_df, 
            'vals', 
            number_of_samples=100,
            show_summary=False,
            return_format='array'
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 100)

    @patch('matplotlib.pyplot.show')
    def test_decimals_bounded(self, mock_show):
        """Test Metalog distribution with bounded decimal numbers."""
        result = CreateMetalogDistribution(
            self.decimal_df, 
            'vals', 
            lower_bound=0, 
            upper_bound=1,
            number_of_samples=50,
            show_summary=False
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 50)
        # Check that samples are within bounds (approximately, as metalog can sometimes slightly vary at tails depending on fit)
        # But logically they should respect bounds if the fit is good.
        self.assertTrue((result['vals'] >= 0).all())
        self.assertTrue((result['vals'] <= 1).all())

    @patch('matplotlib.pyplot.show')
    def test_invalid_return_format(self, mock_show):
        """Test that invalid return_format raises ValueError."""
        with self.assertRaises(ValueError):
            CreateMetalogDistribution(self.int_df, 'vals', return_format='invalid')

if __name__ == '__main__':
    unittest.main()

