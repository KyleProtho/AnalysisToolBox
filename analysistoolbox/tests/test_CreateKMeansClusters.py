import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from analysistoolbox.descriptive_analytics import CreateKMeansClusters


class TestCreateKMeansClusters(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Three well-separated clusters
        cluster1 = pd.DataFrame({'x': np.random.normal(0, 0.5, 40), 'y': np.random.normal(0, 0.5, 40)})
        cluster2 = pd.DataFrame({'x': np.random.normal(5, 0.5, 40), 'y': np.random.normal(5, 0.5, 40)})
        cluster3 = pd.DataFrame({'x': np.random.normal(10, 0.5, 40), 'y': np.random.normal(0, 0.5, 40)})
        self.df = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)

        # DataFrame with a missing value
        self.df_with_nan = self.df.copy()
        self.df_with_nan.loc[0, 'x'] = np.nan

    @patch('matplotlib.pyplot.show')
    def test_returns_dataframe(self, mock_show):
        """Result is a DataFrame with the cluster column appended."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            show_cluster_summary_plots=False,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('K-Means Cluster', result.columns)

    @patch('matplotlib.pyplot.show')
    def test_correct_number_of_clusters(self, mock_show):
        """Cluster column contains exactly the requested number of unique labels."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            show_cluster_summary_plots=False,
        )
        unique_clusters = result['K-Means Cluster'].dropna().unique()
        self.assertEqual(len(unique_clusters), 3)

    @patch('matplotlib.pyplot.show')
    def test_cluster_labels_are_strings(self, mock_show):
        """Cluster labels are stored as strings, not integers."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            show_cluster_summary_plots=False,
        )
        sample_label = result['K-Means Cluster'].dropna().iloc[0]
        self.assertIsInstance(sample_label, str)

    @patch('matplotlib.pyplot.show')
    def test_custom_cluster_column_name(self, mock_show):
        """Custom column name is used for the cluster assignments."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            column_name_for_clusters='Segment',
            show_cluster_summary_plots=False,
        )
        self.assertIn('Segment', result.columns)
        self.assertNotIn('K-Means Cluster', result.columns)

    @patch('matplotlib.pyplot.show')
    def test_row_count_preserved(self, mock_show):
        """Output has the same number of rows as input."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            show_cluster_summary_plots=False,
        )
        self.assertEqual(len(result), len(self.df))

    @patch('matplotlib.pyplot.show')
    def test_missing_values_produce_nan_cluster(self, mock_show):
        """Rows with NaN in clustering columns get NaN in the cluster column."""
        result = CreateKMeansClusters(
            self.df_with_nan,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            show_cluster_summary_plots=False,
        )
        self.assertTrue(result.loc[0, 'K-Means Cluster'] is np.nan or pd.isna(result.loc[0, 'K-Means Cluster']))

    @patch('matplotlib.pyplot.show')
    def test_auto_select_numeric_columns(self, mock_show):
        """When no column list is provided, all numeric columns are used."""
        df = self.df.copy()
        df['label'] = 'cat'  # non-numeric column should be ignored
        result = CreateKMeansClusters(
            df,
            number_of_clusters=3,
            show_cluster_summary_plots=False,
        )
        self.assertIn('K-Means Cluster', result.columns)
        self.assertEqual(len(result), len(df))

    @patch('matplotlib.pyplot.show')
    def test_no_scaling(self, mock_show):
        """Function runs without scaling and still returns valid cluster assignments."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            scale_clustering_column_values=False,
            show_cluster_summary_plots=False,
        )
        self.assertEqual(result['K-Means Cluster'].dropna().nunique(), 3)

    @patch('matplotlib.pyplot.show')
    def test_show_cluster_summary_plots_calls_show(self, mock_show):
        """plt.show() is called once per clustering variable when plots are enabled."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=3,
            show_cluster_summary_plots=True,
        )
        # One plt.show() call per variable (x and y)
        self.assertEqual(mock_show.call_count, 2)

    @patch('matplotlib.pyplot.show')
    def test_elbow_method_when_no_number_of_clusters_specified(self, mock_show):
        """When number_of_clusters is omitted, KElbowVisualizer runs for real and
        its elbow_value_ drives the final cluster count."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            # number_of_clusters intentionally omitted
            show_cluster_summary_plots=False,
        )

        # Result is a valid DataFrame with a cluster column
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('K-Means Cluster', result.columns)
        # Elbow method should detect at least 2 clusters on well-separated data
        n_clusters = result['K-Means Cluster'].dropna().nunique()
        self.assertGreaterEqual(n_clusters, 2)

    @patch('matplotlib.pyplot.show')
    def test_two_cluster_solution(self, mock_show):
        """Function works correctly when exactly 2 clusters are requested."""
        result = CreateKMeansClusters(
            self.df,
            list_of_value_columns_for_clustering=['x', 'y'],
            number_of_clusters=2,
            show_cluster_summary_plots=False,
        )
        self.assertEqual(result['K-Means Cluster'].dropna().nunique(), 2)

    def tearDown(self):
        plt.clf()


if __name__ == '__main__':
    unittest.main()
