import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysistoolbox.visualizations import PlotBoxWhiskerByGroup

class TestPlotBoxWhiskerByGroup(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing
        self.dataframe = pd.DataFrame({
            'Group1': ['A', 'A', 'B', 'B', 'C', 'C'],
            'Group2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'Value': [1, 2, 3, 4, 5, 6]
        })
    
    def test_plot_box_whisker_by_group(self):
        # Test basic box and whisker plot
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1')
        
        # Test box and whisker plot with two grouping variables
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', 'Group2')
        
        # Test box and whisker plot with custom fill color
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', fill_color='blue')
        
        # Test box and whisker plot with custom color palette
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', color_palette='Set3')
        
        # Test box and whisker plot with custom display order
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', display_order_list=['B', 'A', 'C'])
        
        # Test box and whisker plot without legend
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', show_legend=False)
        
        # Test box and whisker plot with custom title and subtitle
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', title_for_plot='Box Plot', subtitle_for_plot='Grouped by Group1')
        
        # Test box and whisker plot with caption and data source
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', caption_for_plot='Custom Caption', data_source_for_plot='Custom Data Source')
        
        # Test box and whisker plot with y-axis
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', show_y_axis=True)
        
        # Test box and whisker plot with custom figure size
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', figure_size=(10, 7))
        
        # Test box and whisker plot with saving the plot
        PlotBoxWhiskerByGroup(self.dataframe, 'Value', 'Group1', filepath_to_save_plot='box_plot.png')
        
        # Add more test cases as needed
        
if __name__ == '__main__':
    unittest.main()