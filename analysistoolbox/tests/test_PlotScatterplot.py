# Import packages
import unittest
import numpy as np
import pandas as pd
from analysistoolbox.visualizations import PlotScatterplot

class TestPlotScatterplot(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe for testing
        self.df = pd.DataFrame({
            'x': np.random.rand(50),
            'y': np.random.rand(50),
            'category': np.random.choice(['Group A', 'Group B'], 50),
            'size': np.random.rand(50) * 100
        })

    def test_basic_scatterplot(self):
        """Test basic scatterplot with no extra options."""
        PlotScatterplot(self.df, 'y', 'x', title_for_plot="Basic Scatterplot")

    def test_grouping_scatterplot(self):
        """Test scatterplot with grouping by a categorical column."""
        PlotScatterplot(self.df, 'y', 'x', grouping_column_name='category', 
                        title_for_plot="Grouped Scatterplot")

    def test_sized_scatterplot(self):
        """Test scatterplot where dot size is determined by a column."""
        PlotScatterplot(self.df, 'y', 'x', size_by_column_name='size', 
                        title_for_plot="Sized Scatterplot")

    def test_fitted_line_none(self):
        """Test scatterplot with fitted_line_type=None (explicitly)."""
        PlotScatterplot(self.df, 'y', 'x', fitted_line_type=None, 
                        title_for_plot="No Fitted Line")

    def test_fitted_line_straight(self):
        """Test scatterplot with a straight regression line."""
        PlotScatterplot(self.df, 'y', 'x', fitted_line_type='straight', 
                        title_for_plot="Straight Regression Line")
        
    def test_fitted_line_straight_grouped(self):
        """Test grouped scatterplot with straight regression lines."""
        PlotScatterplot(self.df, 'y', 'x', grouping_column_name='category', 
                        fitted_line_type='straight', 
                        title_for_plot="Grouped Straight Regression")

    def test_fitted_line_lowess(self):
        """Test scatterplot with a LOWESS smoothing line."""
        PlotScatterplot(self.df, 'y', 'x', fitted_line_type='lowess', 
                        title_for_plot="LOWESS Smoothing Line")

    def test_fitted_line_lowess_grouped(self):
        """Test grouped scatterplot with LOWESS smoothing lines."""
        PlotScatterplot(self.df, 'y', 'x', grouping_column_name='category', 
                        fitted_line_type='lowess', 
                        title_for_plot="Grouped LOWESS Smoothing")

    def test_quadrants(self):
        """Test scatterplot with quadrant labels and fills."""
        PlotScatterplot(
            self.df, 'y', 'x',
            upper_left_quadrant_label="Q1", upper_left_quadrant_fill_color="#ffcccc",
            upper_right_quadrant_label="Q2", upper_right_quadrant_fill_color="#ccffcc",
            lower_left_quadrant_label="Q3", lower_left_quadrant_fill_color="#ccccff",
            lower_right_quadrant_label="Q4", lower_right_quadrant_fill_color="#ffffcc",
            title_for_plot="Quadrant Analysis"
        )

    def test_invalid_line_type(self):
        """Test that invalid fitted_line_type raises a ValueError."""
        with self.assertRaises(ValueError):
            PlotScatterplot(self.df, 'y', 'x', fitted_line_type='invalid_type')

if __name__ == '__main__':
    unittest.main()
