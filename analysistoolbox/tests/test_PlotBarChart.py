import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysistoolbox.visualizations import PlotBarChart

class TestPlotBarChart(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing
        self.dataframe = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D'],
            'Value': [1, 3, 6, 8]
        })
    
    def test_plot_bar_chart(self):
        # Test basic function plot
        PlotBarChart(self.dataframe, 'Category', 'Value')
        
        # Test function plot with custom color palette
        PlotBarChart(self.dataframe, 'Category', 'Value', color_palette='Set2')
        
        # Test function plot with top n categories highlighted
        PlotBarChart(self.dataframe, 'Category', 'Value', top_n_to_highlight=2)
        
        # Test function plot with custom fill color
        PlotBarChart(self.dataframe, 'Category', 'Value', fill_color='blue')
        
        # Test function plot with custom figure size
        PlotBarChart(self.dataframe, 'Category', 'Value', figure_size=(10, 7))
        
        # Test function plot with custom title and subtitle
        PlotBarChart(self.dataframe, 'Category', 'Value', title_for_plot='Custom Title', subtitle_for_plot='Custom Subtitle')
        
        # Test function plot with caption and data source
        PlotBarChart(self.dataframe, 'Category', 'Value', caption_for_plot='Custom Caption', data_source_for_plot='Custom Data Source')
        
        # Test function plot with saving the plot to a file
        PlotBarChart(self.dataframe, 'Category', 'Value', filepath_to_save_plot='plot.png')
        
        # Assert that the plot is displayed
        self.assertTrue(plt.gcf().number > 0)
        
    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()