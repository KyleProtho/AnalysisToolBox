# Load packages
import networkx as nx
import numpy as np
import pandas as pd

# Import local functions
from .BuildGraphFromEdgeList import BuildGraphFromEdgeList

# Centrality measures that treat edge weight as connection STRENGTH: a bigger
# weight makes a node more central (e.g., interaction counts, transaction volume).
_STRENGTH_LIKE_MEASURES = {'degree', 'in_degree', 'out_degree', 'eigenvector', 'pagerank'}

# Centrality measures that treat edge weight as a DISTANCE/cost: a bigger weight
# makes a node less central, because networkx routes shortest paths around it.
_DISTANCE_LIKE_MEASURES = {'betweenness', 'closeness', 'harmonic'}

_VALID_MEASURES = _STRENGTH_LIKE_MEASURES | _DISTANCE_LIKE_MEASURES

# Declare function
def CalculateNodeCentralityMeasures(dataframe,
                                     node_1_column,
                                     node_2_column,
                                     is_directed=False,
                                     edge_weight_column=None,
                                     node_attributes_dataframe=None,
                                     node_id_column=None,
                                     allow_self_loops=False,
                                     allow_multi_edges=False,
                                     list_of_centrality_measures=None,
                                     print_summary=True):
    """
    Calculate one or more node centrality measures from a long-format edge
    list DataFrame, returning one row per node with each measure -- and its
    rank among all nodes -- as a column.

    This function wraps BuildGraphFromEdgeList internally, so analysts pass
    the same raw edge DataFrame arguments they would use to build the graph
    directly; there is no need to construct a networkx graph object first.
    Any node metadata supplied via node_attributes_dataframe (name, type,
    org, country, etc.) is carried through into the output automatically.

    Teaching Note
    -------------
    "Centrality" is not one thing -- it is a family of measures that each
    define "important node" differently, and picking the wrong one for the
    question at hand is a common analytical mistake:
      * Degree centrality (in-degree/out-degree for directed graphs) counts
        direct connections. It answers "who is locally active or popular?"
        but says nothing about a node's position in the broader network.
      * Betweenness centrality counts how often a node sits on the shortest
        path between other pairs of nodes. It answers "who is a broker or
        bottleneck?" -- high-betweenness nodes can control or disrupt the
        flow of information, goods, or disease through a network even if
        they have few direct connections themselves.
      * Closeness centrality measures how few steps it takes to reach every
        other node. It answers "who can spread something fastest?"
      * Harmonic centrality is a variant of closeness that handles
        disconnected graphs gracefully (unreachable nodes contribute zero
        instead of an undefined infinite distance), which makes it more
        robust than closeness on real-world, fragmented networks.
      * Eigenvector centrality and PageRank both weigh a connection by the
        importance of the node on the other end, so being linked to a few
        highly-connected nodes counts for more than being linked to many
        peripheral ones. PageRank additionally dampens the effect of very
        high out-degree nodes flooding their neighbors with influence,
        which is why it tends to be more stable on directed graphs.

    Reporting several of these side by side -- and ranking nodes on each --
    is standard practice, because a node that looks unremarkable on one
    measure (e.g., low degree) can be critical on another (e.g., high
    betweenness), and that mismatch is itself often the interesting finding
    (for example, a low-profile intermediary that many transactions quietly
    route through).

    Weighted graphs: strength vs. distance (a common trap)
    --------------------------------------------------------
    When edge_weight_column is set, networkx does NOT treat "weight" the
    same way across every measure, and this asymmetry silently produces
    misleading results if you don't account for it:
      * Degree, eigenvector centrality, and PageRank treat a bigger weight
        as a STRONGER connection. Degree centrality itself is replaced with
        weighted degree, known as "strength" in network science: the sum of
        a node's incident edge weights rather than a simple edge count. A
        node with three heavily-weighted edges will out-rank a node with
        ten barely-weighted ones.
      * Betweenness, closeness, and harmonic centrality treat a bigger
        weight as a LONGER/COSTLIER path, because they are built on
        shortest-path algorithms where "weight" means "distance". A node
        connected by high-weight edges will look FARTHER away on these
        measures, not closer.
    If your edge_weight_column represents something strength-like (call
    volume, dollars transacted, number of shared meetings), betweenness,
    closeness, and harmonic centrality computed directly on it will treat
    your most active relationships as the least traversable ones. To get
    sensible path-based results in that case, invert the weight before
    calling this function (e.g., 1 / weight, or max(weight) - weight + 1)
    so that "more interaction" maps to "shorter distance". This function
    prints a reminder of which requested measures fall on which side of
    this divide whenever a weighted graph is used.

    Parameters
    ----------
    dataframe
        A pandas DataFrame in long format, with one row per edge (relationship).
    node_1_column
        Name of the column containing the first endpoint of each edge.
    node_2_column
        Name of the column containing the second endpoint of each edge.
    is_directed
        Whether the relationship is directional (node_1 -> node_2). Defaults
        to False. When True, 'in_degree' and 'out_degree' become available
        (and are used in place of 'degree' by default); when False, only the
        undirected 'degree' measure is available.
    edge_weight_column
        Optional name of a column containing edge weights, attached to each
        edge as a 'weight' attribute. See "Weighted graphs" above for how
        this changes each measure's interpretation. Defaults to None.
    node_attributes_dataframe
        Optional DataFrame of node-level metadata (e.g., name, type,
        organization, country) to attach to nodes and carry through into the
        output. Must be used together with node_id_column. Defaults to None.
    node_id_column
        Name of the column in node_attributes_dataframe that identifies each
        node. Required if node_attributes_dataframe is provided. Defaults to
        None.
    allow_self_loops
        Whether to keep edges where node_1 and node_2 are the same entity.
        Defaults to False. See BuildGraphFromEdgeList for details.
    allow_multi_edges
        Must be False. Centrality algorithms in networkx are not defined for
        MultiGraph/MultiDiGraph, so multi-edges cannot be passed through to
        this function. If your data has parallel edges, aggregate them into
        an edge_weight_column (e.g., a count or sum) before calling this
        function. Defaults to False.
    list_of_centrality_measures
        List of centrality measures to calculate. Valid values are 'degree',
        'in_degree', 'out_degree' (directed graphs only), 'betweenness',
        'closeness', 'eigenvector', 'pagerank', and 'harmonic' -- each maps
        directly to the corresponding algorithm documented at
        https://networkx.org/documentation/stable/reference/algorithms/centrality.html.
        If None, defaults to ['degree', 'betweenness', 'closeness',
        'eigenvector', 'pagerank', 'harmonic'] for undirected graphs, or
        ['in_degree', 'out_degree', 'betweenness', 'closeness', 'eigenvector',
        'pagerank', 'harmonic'] for directed graphs. Defaults to None.
    print_summary
        Whether to print the graph build report (see BuildGraphFromEdgeList),
        the weighted-measure semantics reminder (if applicable), and a final
        count of measures calculated. Defaults to True.

    Returns
    -------
    pd.DataFrame
        One row per node, containing:
          * 'Node' -- the node identifier.
          * Any columns carried over from node_attributes_dataframe.
          * One column per requested centrality measure (e.g.,
            'Degree_Centrality', 'Degree_Strength' if weighted,
            'Betweenness_Centrality', 'PageRank', etc.).
          * One 'Rank_<measure column>' column per measure, ranking nodes
            from most central (1) to least central, with ties sharing the
            same rank (method='min').

    Examples
    --------
    # Unweighted, undirected: who brokers connections between clusters?
    import pandas as pd
    edges_df = pd.DataFrame({
        'person_a': ['Alice', 'Alice', 'Bob', 'Carol', 'Carol', 'Dave'],
        'person_b': ['Bob', 'Carol', 'Carol', 'Dave', 'Eve', 'Eve']
    })
    centrality_df = CalculateNodeCentralityMeasures(
        edges_df,
        node_1_column='person_a',
        node_2_column='person_b',
        list_of_centrality_measures=['degree', 'betweenness']
    )
    top_broker = centrality_df.sort_values('Rank_Betweenness_Centrality').iloc[0]

    # Weighted, directed: transaction network with node metadata attached
    node_lookup_df = pd.DataFrame({
        'account_id': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
        'account_type': ['individual', 'individual', 'business', 'business', 'individual']
    })
    weighted_df = CalculateNodeCentralityMeasures(
        edges_df.assign(dollar_amount=[500, 1200, 300, 8000, 150, 4200]),
        node_1_column='person_a',
        node_2_column='person_b',
        is_directed=True,
        edge_weight_column='dollar_amount',
        node_attributes_dataframe=node_lookup_df,
        node_id_column='account_id',
        list_of_centrality_measures=['in_degree', 'out_degree', 'pagerank']
    )
    """
    # Multi-edges are not supported by networkx centrality algorithms
    if allow_multi_edges:
        raise ValueError(
            "CalculateNodeCentralityMeasures does not support allow_multi_edges=True: "
            "networkx centrality algorithms are not defined for MultiGraph/MultiDiGraph. "
            "Aggregate parallel edges into an edge_weight_column (e.g., a count or sum) "
            "before calling this function."
        )

    # Determine the default measures based on directedness, if none were specified
    if list_of_centrality_measures is None:
        if is_directed:
            list_of_centrality_measures = ['in_degree', 'out_degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'harmonic']
        else:
            list_of_centrality_measures = ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'harmonic']

    # Validate the requested measures
    unknown_measures = set(list_of_centrality_measures) - _VALID_MEASURES
    if unknown_measures:
        raise ValueError(
            f"Unknown centrality measure(s) {sorted(unknown_measures)}. Valid options are "
            f"{sorted(_VALID_MEASURES)}, matching the algorithms documented at "
            "https://networkx.org/documentation/stable/reference/algorithms/centrality.html"
        )
    if not is_directed:
        directed_only_requested = {'in_degree', 'out_degree'} & set(list_of_centrality_measures)
        if directed_only_requested:
            raise ValueError(
                f"{sorted(directed_only_requested)} require is_directed=True; "
                "in-degree and out-degree centrality are undefined for undirected graphs."
            )

    # Build the graph from the raw edge DataFrame
    graph = BuildGraphFromEdgeList(
        dataframe,
        node_1_column=node_1_column,
        node_2_column=node_2_column,
        is_directed=is_directed,
        edge_weight_column=edge_weight_column,
        node_attributes_dataframe=node_attributes_dataframe,
        node_id_column=node_id_column,
        allow_self_loops=allow_self_loops,
        allow_multi_edges=allow_multi_edges,
        print_summary=print_summary
    )
    is_weighted = edge_weight_column is not None
    weight_argument = 'weight' if is_weighted else None

    # Start the results frame with one row per node
    results = pd.DataFrame({'Node': list(graph.nodes())})

    # Carry through any node attributes attached to the graph (e.g., from
    # node_attributes_dataframe), so downstream reporting doesn't need to
    # re-join the lookup table
    node_attributes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    if len(node_attributes.columns) > 0:
        node_attributes.index.name = 'Node'
        node_attributes = node_attributes.reset_index()
        results = results.merge(node_attributes, on='Node', how='left')

    # Calculate each requested centrality measure
    for measure in list_of_centrality_measures:
        column_name, values = _CalculateSingleCentralityMeasure(graph, measure, is_weighted, weight_argument)
        results[column_name] = results['Node'].map(values)
        results[f'Rank_{column_name}'] = results[column_name].rank(method='min', ascending=False).astype('Int64')

    # Print a report on the weighted-measure semantics and what was calculated
    if print_summary:
        if is_weighted:
            strength_like = [measure for measure in list_of_centrality_measures if measure in _STRENGTH_LIKE_MEASURES]
            distance_like = [measure for measure in list_of_centrality_measures if measure in _DISTANCE_LIKE_MEASURES]
            print(f"\nWeighted graph note (edge_weight_column='{edge_weight_column}'):")
            if strength_like:
                print(f"  Treated as STRENGTH (higher = more central): {', '.join(strength_like)}")
            if distance_like:
                print(f"  Treated as DISTANCE (higher = less central): {', '.join(distance_like)}")
                print("  If your weights represent strength (e.g., interaction counts), invert them before")
                print("  calling this function so these path-based measures behave as expected.")
        print(f"\nCalculated {len(list_of_centrality_measures)} centrality measure(s) for {graph.number_of_nodes()} nodes.")

    # Return the results
    return results


def _CalculateSingleCentralityMeasure(graph, measure, is_weighted, weight_argument):
    """
    Internal helper that dispatches to the appropriate networkx centrality
    algorithm for a single measure and returns (column_name, values_dict).
    """
    if measure == 'degree':
        if is_weighted:
            return 'Degree_Strength', dict(graph.degree(weight=weight_argument))
        return 'Degree_Centrality', nx.degree_centrality(graph)

    if measure == 'in_degree':
        if is_weighted:
            return 'In_Degree_Strength', dict(graph.in_degree(weight=weight_argument))
        return 'In_Degree_Centrality', nx.in_degree_centrality(graph)

    if measure == 'out_degree':
        if is_weighted:
            return 'Out_Degree_Strength', dict(graph.out_degree(weight=weight_argument))
        return 'Out_Degree_Centrality', nx.out_degree_centrality(graph)

    if measure == 'betweenness':
        return 'Betweenness_Centrality', nx.betweenness_centrality(graph, weight=weight_argument)

    if measure == 'closeness':
        return 'Closeness_Centrality', nx.closeness_centrality(graph, distance=weight_argument)

    if measure == 'eigenvector':
        return 'Eigenvector_Centrality', _CalculateEigenvectorCentralityWithFallback(graph, weight_argument)

    if measure == 'pagerank':
        return 'PageRank', nx.pagerank(graph, weight=weight_argument)

    if measure == 'harmonic':
        return 'Harmonic_Centrality', nx.harmonic_centrality(graph, distance=weight_argument)

    raise ValueError(f"Unhandled centrality measure '{measure}'.")


def _CalculateEigenvectorCentralityWithFallback(graph, weight_argument):
    """
    Internal helper that calculates eigenvector centrality, falling back to
    the numpy-based solver, and finally to NaN, if the power iteration does
    not converge. Eigenvector centrality is undefined (or numerically
    unstable) on graphs that are disconnected, directed-but-not-strongly-
    connected, or have isolated nodes, so a clean failure mode matters more
    than raising and aborting the whole calculation.
    """
    if graph.number_of_nodes() == 0:
        return {}

    try:
        return nx.eigenvector_centrality(graph, weight=weight_argument, max_iter=1000)
    except nx.NetworkXException:
        pass

    try:
        return nx.eigenvector_centrality_numpy(graph, weight=weight_argument)
    except Exception:
        print(
            "Warning: eigenvector centrality did not converge and was set to NaN for all nodes. "
            "This commonly happens on disconnected graphs, or directed graphs that are not "
            "strongly connected."
        )
        return {node: np.nan for node in graph.nodes()}
