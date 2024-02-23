import unittest
import pandas as pd
import matplotlib.pyplot as plt
from analysistoolbox.visualizations import PlotBulletChart

class TestPlotBulletChart(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing
        self.dataframe = pd.DataFrame({
            'Group': ['A', 'B', 'C', 'D'],
            'Value': [1, 3, 5, 8],
            'Target': [2, 4, 6, 7],
            'Limit1': [0, 2, 4, 6],
            'Limit2': [2, 4, 6, 8],
            'Limit3': [1, 3, 5, 7],
        })
    
    def test_plot_bullet_chart(self):
        # Test basic function plot
        PlotBulletChart(self.dataframe, 'Value', 'Group')
        
        # Test function plot with target value
        PlotBulletChart(self.dataframe, 'Value', 'Group', target_value_column_name='Target')
        
        # Test function plot with limit columns
        PlotBulletChart(self.dataframe, 'Value', 'Group', list_of_limit_columns=['Limit1', 'Limit2'])
        
        # Test function plot with custom value range
        PlotBulletChart(self.dataframe, 'Value', 'Group', value_minimum=0, value_maximum=10)
        
        # Test function plot with custom display order
        PlotBulletChart(self.dataframe, 'Value', 'Group', display_order_list=['C', 'B', 'A', 'D'])
        
        # Test function plot with custom colors
        PlotBulletChart(self.dataframe, 'Value', 'Group', background_color_palette='Set2', value_dot_color='red')
        
        # Test function plot with saving the plot to a file
        PlotBulletChart(self.dataframe, 'Value', 'Group', filepath_to_save_plot='plot.png')
        
        # Assert that the plot is displayed
        self.assertTrue(plt.gcf().number > 0)
        
    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()