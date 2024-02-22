import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysistoolbox.visualizations import PlotClusteredBarChart

class TestPlotClusteredBarChart(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing
        self.dataframe = pd.DataFrame({
            'Group1': ['A', 'A', 'B', 'B', 'C', 'C', 'C', 'D'],
            'Group2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'Z', 'Z'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8]
        })
    
    def test_plot_clustered_bar_chart(self):
        # Test basic function plot
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value')
        
        # Test function plot with custom color palette
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', color_palette='Set2')
        
        # Test function plot with custom fill transparency
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', fill_transparency=0.5)
        
        # Test function plot with custom display order list
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', display_order_list=['C', 'B', 'A'])
        
        # Test function plot with custom figure size
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', figure_size=(8, 6))
        
        # Test function plot with legend hidden
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', show_legend=False)
        
        # Test function plot with decimal places for data labels
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', decimal_places_for_data_label=2)
        
        # Test function plot with custom title and subtitle
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', title_for_plot='Custom Title', subtitle_for_plot='Custom Subtitle')
        
        # Test function plot with caption and data source
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', caption_for_plot='Custom Caption', data_source_for_plot='Custom Data Source')
        
        # Test function plot with saving the plot to a file
        PlotClusteredBarChart(self.dataframe, 'Group1', 'Group2', 'Value', filepath_to_save_plot='plot.png')
        
        # Assert that the plot is displayed
        self.assertTrue(plt.gcf().number > 0)
        
    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()