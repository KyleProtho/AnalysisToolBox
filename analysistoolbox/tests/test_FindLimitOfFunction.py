# test_FindLimitOfFunction.py
import unittest
from analysistoolbox.calculus import FindLimitOfFunction

class TestFindLimitOfFunction(unittest.TestCase):
    def test_linear_function(self):
        # Define a simple linear function
        f_of_x = lambda x: 2*x + 3

        # Test the function at the point x = 1
        limit = FindLimitOfFunction(f_of_x, 1, plot_function=False, plot_tangent_line=False)

        # The limit of a linear function at any point should be the function value at that point
        self.assertEqual(limit, f_of_x(1))

if __name__ == '__main__':
    unittest.main()
