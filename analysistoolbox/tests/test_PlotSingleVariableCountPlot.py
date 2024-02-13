import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysistoolbox.visualizations import PlotSingleVariableCountPlot

class TestPlotSingleVariableCountPlot(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing
        self.dataframe = pd.DataFrame({
            'Category': ['A', 'A', 'B', 'B', 'C', 'C', 'C', 'D'],
            'Value': [1, 2, 3, 4, 5, 6, 7, 8]
        })
    
    def test_plot_single_variable_count_plot(self):
        # Test basic function plot
        PlotSingleVariableCountPlot(self.dataframe, 'Category')
        
        # Test function plot with custom color palette
        PlotSingleVariableCountPlot(self.dataframe, 'Category', color_palette='Set2')
        
        # Test function plot with top n categories highlighted
        PlotSingleVariableCountPlot(self.dataframe, 'Category', top_n_to_highlight=2)
        
        # Test function plot with custom fill color
        PlotSingleVariableCountPlot(self.dataframe, 'Category', fill_color='blue')
        
        # Test function plot with rare category line
        PlotSingleVariableCountPlot(self.dataframe, 'Category', add_rare_category_line=True)
        
        # Test function plot with custom figure size
        PlotSingleVariableCountPlot(self.dataframe, 'Category', figure_size=(10, 7))
        
        # Test function plot with custom title and subtitle
        PlotSingleVariableCountPlot(self.dataframe, 'Category', title_for_plot='Custom Title', subtitle_for_plot='Custom Subtitle')
        
        # Test function plot with caption and data source
        PlotSingleVariableCountPlot(self.dataframe, 'Category', caption_for_plot='Custom Caption', data_source_for_plot='Custom Data Source')
        
        # Test function plot with saving the plot to a file
        PlotSingleVariableCountPlot(self.dataframe, 'Category', filepath_to_save_plot='plot.png')
        
        # Assert that the plot is displayed
        self.assertTrue(plt.gcf().number > 0)
        
    def tearDown(self):
        # Clear the plot
        plt.clf()

if __name__ == '__main__':
    unittest.main()