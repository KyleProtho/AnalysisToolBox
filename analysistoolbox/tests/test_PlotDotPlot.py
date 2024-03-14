import unittest
import pandas as pd
import matplotlib.pyplot as plt
from analysistoolbox.visualizations import PlotDotPlot

class TestPlotDotPlot(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing
        self.dataframe = pd.DataFrame({
            'Category': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8],
            'Group': ['Group 1', 'Group 2', 'Group 1', 'Group 2', 'Group 1', 'Group 2', 'Group 1', 'Group 2']
        })
    
    def test_plot_dot_plot(self):
        # Test basic function plot
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group')
        
        # Test function plot with custom dot size and alpha
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', dot_size=10, dot_alpha=0.5)
        
        # Test function plot with connecting lines disabled
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', connect_dots=False)
        
        # Test function plot with custom line color and style
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', connect_line_color='red', connect_line_style='solid')
        
        # Test function plot with custom line labels
        line_labels = {'A': 'Label A', 'B': 'Label B'}
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', dict_of_connect_line_labels=line_labels)
        
        # Test function plot with zero line group
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', zero_line_group='Group 1')
        
        # Test function plot with custom display order
        display_order = ['C', 'B', 'A', 'D']
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', display_order_list=display_order)
        
        # Test function plot with data labels disabled
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', show_data_labels=False)
        
        # Test function plot with custom title and subtitle
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', title_for_plot='Custom Title', subtitle_for_plot='Custom Subtitle')
        
        # Test function plot with caption and data source
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', caption_for_plot='Custom Caption', data_source_for_plot='Custom Data Source')
        
        # Test function plot with saving the plot to a file
        PlotDotPlot(self.dataframe, 'Category', 'Value', 'Group', filepath_to_save_plot='plot.png')
        
        # Assert that the plot is displayed
        self.assertTrue(plt.gcf().number > 0)
        
    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()