# test_PlotFunction.py
import unittest
import numpy as np
import sympy
from analysistoolbox.calculus import PlotFunction

class TestPlotFunction(unittest.TestCase):
    
    def test_plot_function(self):
        # Set sympy symbol
        x = sympy.Symbol('x')

        # Test basic function plot
        PlotFunction(lambda x: x**2)
        
        # Test function plot with custom x range
        PlotFunction(lambda x: x**2, minimum_x=-5, maximum_x=5)
        
        # Test function plot with custom number of points
        PlotFunction(lambda x: x**2, n=50)
        
        # Test function plot with custom line color
        PlotFunction(lambda x: x**2, line_color="red")
        
        # Test function plot with custom marker style
        PlotFunction(lambda x: x**2, markers="x")
        
        # Test function plot with custom axis labels
        PlotFunction(lambda x: x**2, x_axis_variable_name="x-axis", y_axis_variable_name="y-axis")
        
        # Test function plot with custom title and subtitle
        PlotFunction(lambda x: x**2, title_for_plot="Custom Title", subtitle_for_plot="Custom Subtitle")
        
        # Test function plot with caption and data source
        PlotFunction(lambda x: x**2, caption_for_plot="Custom Caption", data_source_for_plot="Custom Data Source")
        
        # Test function plot with custom figure size
        PlotFunction(lambda x: x**2, figure_size=(10, 7))
        
if __name__ == '__main__':
    unittest.main()
