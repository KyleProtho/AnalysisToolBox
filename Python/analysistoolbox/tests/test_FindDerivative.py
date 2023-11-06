# test_FindDerivative.py
import unittest
from analysistoolbox.calculus import FindDerivative

class TestFindDerivative(unittest.TestCase):
    def test_find_derivative(self):
        # Define a simple function for testing
        def f(x):
            return x**2

        # Define its derivative
        def df(x):
            return 2*x

        # Get the derivative function from your function
        derivative_function = FindDerivative(f, return_derivative_function=True)

        # Test the derivative function at a few points
        for x in [-2, -1, 0, 1, 2]:
            self.assertAlmostEqual(derivative_function(x), df(x), places=5)

if __name__ == '__main__':
    unittest.main()
