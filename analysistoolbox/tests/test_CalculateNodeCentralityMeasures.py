import unittest
import unittest.mock
import networkx as nx
import numpy as np
import pandas as pd
from analysistoolbox.network_analysis import CalculateNodeCentralityMeasures


class TestCalculateNodeCentralityMeasures(unittest.TestCase):

    def setUp(self):
        # Path graph: Alice-Bob-Carol-Dave-Eve, with Carol as the clear broker
        self.edges_df = pd.DataFrame({
            'person_a': ['Alice', 'Bob', 'Carol', 'Dave'],
            'person_b': ['Bob', 'Carol', 'Dave', 'Eve'],
        })
        self.weighted_edges_df = self.edges_df.assign(amount=[500, 1200, 8000, 150])
        self.node_attributes_df = pd.DataFrame({
            'entity_id': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
            'org': ['Acme', 'Acme', 'Globex', 'Globex', 'Initech'],
        })

    def test_default_measures_undirected(self):
        result = CalculateNodeCentralityMeasures(
            self.edges_df, 'person_a', 'person_b', print_summary=False
        )
        for column in ['Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality',
                        'Eigenvector_Centrality', 'PageRank', 'Harmonic_Centrality']:
            self.assertIn(column, result.columns)
            self.assertIn(f'Rank_{column}', result.columns)
        self.assertEqual(len(result), 5)

    def test_default_measures_directed(self):
        result = CalculateNodeCentralityMeasures(
            self.edges_df, 'person_a', 'person_b', is_directed=True, print_summary=False
        )
        self.assertIn('In_Degree_Centrality', result.columns)
        self.assertIn('Out_Degree_Centrality', result.columns)
        self.assertNotIn('Degree_Centrality', result.columns)

    def test_explicit_measures_only_requested_columns_present(self):
        result = CalculateNodeCentralityMeasures(
            self.edges_df, 'person_a', 'person_b',
            list_of_centrality_measures=['degree', 'betweenness'],
            print_summary=False
        )
        self.assertIn('Degree_Centrality', result.columns)
        self.assertIn('Betweenness_Centrality', result.columns)
        self.assertNotIn('PageRank', result.columns)
        self.assertNotIn('Closeness_Centrality', result.columns)

    def test_broker_has_highest_betweenness(self):
        result = CalculateNodeCentralityMeasures(
            self.edges_df, 'person_a', 'person_b',
            list_of_centrality_measures=['betweenness'],
            print_summary=False
        )
        top_node = result.loc[result['Rank_Betweenness_Centrality'] == 1, 'Node'].iloc[0]
        self.assertEqual(top_node, 'Carol')

    def test_rank_column_is_nullable_int(self):
        result = CalculateNodeCentralityMeasures(
            self.edges_df, 'person_a', 'person_b',
            list_of_centrality_measures=['degree'],
            print_summary=False
        )
        self.assertEqual(str(result['Rank_Degree_Centrality'].dtype), 'Int64')

    def test_weighted_degree_becomes_strength(self):
        result = CalculateNodeCentralityMeasures(
            self.weighted_edges_df, 'person_a', 'person_b',
            edge_weight_column='amount',
            list_of_centrality_measures=['degree'],
            print_summary=False
        )
        self.assertIn('Degree_Strength', result.columns)
        self.assertNotIn('Degree_Centrality', result.columns)
        # Carol touches edges worth 1200 + 8000 = 9200
        self.assertEqual(result.loc[result['Node'] == 'Carol', 'Degree_Strength'].iloc[0], 9200)

    def test_weighted_directed_degree_becomes_in_out_strength(self):
        result = CalculateNodeCentralityMeasures(
            self.weighted_edges_df, 'person_a', 'person_b',
            is_directed=True,
            edge_weight_column='amount',
            list_of_centrality_measures=['in_degree', 'out_degree'],
            print_summary=False
        )
        self.assertIn('In_Degree_Strength', result.columns)
        self.assertIn('Out_Degree_Strength', result.columns)
        self.assertEqual(result.loc[result['Node'] == 'Eve', 'In_Degree_Strength'].iloc[0], 150)
        self.assertEqual(result.loc[result['Node'] == 'Alice', 'Out_Degree_Strength'].iloc[0], 500)

    def test_node_attributes_carried_into_output(self):
        result = CalculateNodeCentralityMeasures(
            self.edges_df, 'person_a', 'person_b',
            node_attributes_dataframe=self.node_attributes_df,
            node_id_column='entity_id',
            list_of_centrality_measures=['degree'],
            print_summary=False
        )
        self.assertIn('org', result.columns)
        self.assertEqual(result.loc[result['Node'] == 'Carol', 'org'].iloc[0], 'Globex')

    def test_isolated_node_from_attributes_table_included(self):
        node_attributes_with_isolate = pd.concat([
            self.node_attributes_df,
            pd.DataFrame({'entity_id': ['Frank'], 'org': ['Umbrella']})
        ], ignore_index=True)
        result = CalculateNodeCentralityMeasures(
            self.edges_df, 'person_a', 'person_b',
            node_attributes_dataframe=node_attributes_with_isolate,
            node_id_column='entity_id',
            list_of_centrality_measures=['degree'],
            print_summary=False
        )
        self.assertIn('Frank', result['Node'].values)
        self.assertEqual(result.loc[result['Node'] == 'Frank', 'Degree_Centrality'].iloc[0], 0.0)

    def test_multi_edges_raise_value_error(self):
        with self.assertRaises(ValueError):
            CalculateNodeCentralityMeasures(
                self.edges_df, 'person_a', 'person_b',
                allow_multi_edges=True, print_summary=False
            )

    def test_unknown_measure_raises_value_error(self):
        with self.assertRaises(ValueError):
            CalculateNodeCentralityMeasures(
                self.edges_df, 'person_a', 'person_b',
                list_of_centrality_measures=['not_a_real_measure'],
                print_summary=False
            )

    def test_in_degree_on_undirected_graph_raises_value_error(self):
        with self.assertRaises(ValueError):
            CalculateNodeCentralityMeasures(
                self.edges_df, 'person_a', 'person_b',
                is_directed=False,
                list_of_centrality_measures=['in_degree'],
                print_summary=False
            )

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            CalculateNodeCentralityMeasures(
                self.edges_df, 'nonexistent', 'person_b', print_summary=False
            )

    def test_print_summary_prints_weighted_semantics_note(self):
        with unittest.mock.patch('builtins.print') as mock_print:
            CalculateNodeCentralityMeasures(
                self.weighted_edges_df, 'person_a', 'person_b',
                edge_weight_column='amount',
                list_of_centrality_measures=['pagerank', 'betweenness'],
                print_summary=True
            )
            printed_text = ' '.join(str(call.args[0]) for call in mock_print.call_args_list)
            self.assertIn('STRENGTH', printed_text)
            self.assertIn('DISTANCE', printed_text)

    def test_no_print_when_print_summary_false(self):
        with unittest.mock.patch('builtins.print') as mock_print:
            CalculateNodeCentralityMeasures(
                self.edges_df, 'person_a', 'person_b', print_summary=False
            )
            mock_print.assert_not_called()

    def test_null_graph_does_not_raise(self):
        # All rows are self-loops, so after cleaning the graph has zero nodes
        # and zero edges. Eigenvector centrality has no defined solution for
        # the null graph; the function should return an empty result rather
        # than propagate networkx's NetworkXPointlessConcept error.
        df = pd.DataFrame({'a': ['X', 'Y'], 'b': ['X', 'Y']})
        result = CalculateNodeCentralityMeasures(
            df, 'a', 'b',
            list_of_centrality_measures=['eigenvector'],
            print_summary=False
        )
        self.assertEqual(len(result), 0)
        self.assertIn('Eigenvector_Centrality', result.columns)

    def test_eigenvector_failure_falls_back_to_nan_without_raising(self):
        with unittest.mock.patch(
            'analysistoolbox.network_analysis.CalculateNodeCentralityMeasures.nx.eigenvector_centrality',
            side_effect=nx.PowerIterationFailedConvergence(1000)
        ), unittest.mock.patch(
            'analysistoolbox.network_analysis.CalculateNodeCentralityMeasures.nx.eigenvector_centrality_numpy',
            side_effect=RuntimeError('did not converge')
        ):
            result = CalculateNodeCentralityMeasures(
                self.edges_df, 'person_a', 'person_b',
                list_of_centrality_measures=['eigenvector'],
                print_summary=False
            )
        self.assertTrue(result['Eigenvector_Centrality'].isna().all())


if __name__ == '__main__':
    unittest.main()
