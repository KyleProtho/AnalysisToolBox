import unittest
import unittest.mock
import networkx as nx
import numpy as np
import pandas as pd
from analysistoolbox.network_analysis import BuildGraphFromEdgeList


class TestBuildGraphFromEdgeList(unittest.TestCase):

    def setUp(self):
        self.edges_df = pd.DataFrame({
            'source': ['Alice', 'Alice', 'Bob', 'Carol'],
            'target': ['Bob', 'Carol', 'Carol', 'Carol'],
            'weight': [3, 1, 2, 4],
        })
        self.node_attributes_df = pd.DataFrame({
            'entity_id': ['Alice', 'Bob', 'Carol', 'Dave'],
            'org': ['Acme', 'Acme', 'Globex', 'Globex'],
        })

    def test_returns_undirected_graph_by_default(self):
        graph = BuildGraphFromEdgeList(
            self.edges_df, 'source', 'target', print_summary=False
        )
        self.assertIsInstance(graph, nx.Graph)
        self.assertNotIsInstance(graph, nx.DiGraph)
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 3)

    def test_returns_directed_graph_when_requested(self):
        graph = BuildGraphFromEdgeList(
            self.edges_df, 'source', 'target', is_directed=True, print_summary=False
        )
        self.assertIsInstance(graph, nx.DiGraph)

    def test_edge_weight_attached_as_weight(self):
        graph = BuildGraphFromEdgeList(
            self.edges_df, 'source', 'target',
            edge_weight_column='weight', print_summary=False
        )
        self.assertEqual(graph['Alice']['Bob']['weight'], 3)

    def test_no_weight_attribute_when_not_specified(self):
        graph = BuildGraphFromEdgeList(
            self.edges_df, 'source', 'target', print_summary=False
        )
        self.assertNotIn('weight', graph['Alice']['Bob'])

    def test_node_attributes_attached(self):
        graph = BuildGraphFromEdgeList(
            self.edges_df, 'source', 'target',
            node_attributes_dataframe=self.node_attributes_df,
            node_id_column='entity_id',
            print_summary=False
        )
        self.assertEqual(graph.nodes['Alice']['org'], 'Acme')
        self.assertEqual(graph.nodes['Carol']['org'], 'Globex')

    def test_node_attributes_add_isolated_nodes(self):
        # 'Dave' is in the lookup table but has no edges
        graph = BuildGraphFromEdgeList(
            self.edges_df, 'source', 'target',
            node_attributes_dataframe=self.node_attributes_df,
            node_id_column='entity_id',
            print_summary=False
        )
        self.assertIn('Dave', graph.nodes)
        self.assertEqual(graph.degree('Dave'), 0)

    def test_node_attributes_dataframe_requires_node_id_column(self):
        with self.assertRaises(ValueError):
            BuildGraphFromEdgeList(
                self.edges_df, 'source', 'target',
                node_attributes_dataframe=self.node_attributes_df,
                print_summary=False
            )

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            BuildGraphFromEdgeList(
                self.edges_df, 'nonexistent', 'target', print_summary=False
            )

    def test_self_loops_dropped_by_default(self):
        df = pd.DataFrame({
            'source': ['Alice', 'Bob', 'Carol'],
            'target': ['Alice', 'Carol', 'Bob'],
        })
        graph = BuildGraphFromEdgeList(df, 'source', 'target', print_summary=False)
        self.assertNotIn(('Alice', 'Alice'), graph.edges)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(
            graph.graph['edge_list_build_summary']['self_loops_dropped'], 1
        )

    def test_self_loops_kept_when_allowed(self):
        df = pd.DataFrame({
            'source': ['Alice', 'Bob'],
            'target': ['Alice', 'Carol'],
        })
        graph = BuildGraphFromEdgeList(
            df, 'source', 'target', allow_self_loops=True, print_summary=False
        )
        self.assertTrue(graph.has_edge('Alice', 'Alice'))
        self.assertEqual(
            graph.graph['edge_list_build_summary']['self_loops_dropped'], 0
        )

    def test_multi_edges_collapsed_by_default(self):
        df = pd.DataFrame({
            'source': ['Alice', 'Bob'],
            'target': ['Bob', 'Alice'],  # same undirected pair, reversed
        })
        graph = BuildGraphFromEdgeList(df, 'source', 'target', print_summary=False)
        self.assertIsInstance(graph, nx.Graph)
        self.assertNotIsInstance(graph, nx.MultiGraph)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(
            graph.graph['edge_list_build_summary']['multi_edges_dropped'], 1
        )

    def test_multi_edges_kept_when_allowed_builds_multigraph(self):
        df = pd.DataFrame({
            'source': ['Alice', 'Bob'],
            'target': ['Bob', 'Alice'],
        })
        graph = BuildGraphFromEdgeList(
            df, 'source', 'target', allow_multi_edges=True, print_summary=False
        )
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(
            graph.graph['edge_list_build_summary']['multi_edges_dropped'], 0
        )

    def test_directed_multi_edges_not_collapsed_across_direction(self):
        # Directed graphs treat (Alice, Bob) and (Bob, Alice) as distinct edges
        df = pd.DataFrame({
            'source': ['Alice', 'Bob'],
            'target': ['Bob', 'Alice'],
        })
        graph = BuildGraphFromEdgeList(
            df, 'source', 'target', is_directed=True, print_summary=False
        )
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(
            graph.graph['edge_list_build_summary']['multi_edges_dropped'], 0
        )

    def test_rows_with_missing_endpoints_dropped(self):
        df = pd.DataFrame({
            'source': ['Alice', np.nan],
            'target': ['Bob', 'Carol'],
        })
        graph = BuildGraphFromEdgeList(df, 'source', 'target', print_summary=False)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(
            graph.graph['edge_list_build_summary']['rows_dropped_for_missing_endpoints'], 1
        )

    def test_build_summary_recorded_on_graph(self):
        graph = BuildGraphFromEdgeList(
            self.edges_df, 'source', 'target', print_summary=False
        )
        summary = graph.graph['edge_list_build_summary']
        self.assertEqual(summary['node_count'], graph.number_of_nodes())
        self.assertEqual(summary['edge_count'], graph.number_of_edges())

    def test_print_summary_prints_report(self):
        with unittest.mock.patch('builtins.print') as mock_print:
            BuildGraphFromEdgeList(self.edges_df, 'source', 'target', print_summary=True)
            self.assertTrue(mock_print.called)


if __name__ == '__main__':
    unittest.main()
