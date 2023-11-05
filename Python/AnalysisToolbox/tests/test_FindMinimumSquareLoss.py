# test_FindMinimumSquareLoss.py
import unittest
import numpy as np
from AnalysisToolbox.Calculus import FindMinimumSquareLoss

class TestFindMinimumSquareLoss(unittest.TestCase):
    def test_equal_length_inputs(self):
        observed_values = [1, 2, 3, 4, 5]
        predicted_values = [1, 2, 3, 4, 5]
        
        # Test the function with equal length inputs
        result = FindMinimumSquareLoss(observed_values, predicted_values, show_plot=False)
        
        # The minimum square loss of equal values should be 0
        self.assertEqual(result, 0)

    def test_unequal_length_inputs(self):
        observed_values = [1, 2, 3, 4, 5]
        predicted_values = [1, 2, 3, 4]
        
        # Test the function with unequal length inputs
        with self.assertRaises(ValueError):
            FindMinimumSquareLoss(observed_values, predicted_values, show_plot=False)

    def test_numpy_array_inputs(self):
        observed_values = np.array([1, 2, 3, 4, 5])
        predicted_values = np.array([1, 2, 3, 4, 5])
        
        # Test the function with numpy array inputs
        result = FindMinimumSquareLoss(observed_values, predicted_values, show_plot=False)
        
        # The minimum square loss of equal values should be 0
        self.assertEqual(result, 0)

    def test_different_values(self):
        observed_values = [1, 2, 3, 4, 5]
        predicted_values = [2, 3, 4, 5, 6]
        
        # Test the function with different values
        result = FindMinimumSquareLoss(observed_values, predicted_values, show_plot=False)
        
        # The minimum square loss should be 1
        self.assertEqual(result, 1)

if __name__ == '__main__':
    unittest.main()
