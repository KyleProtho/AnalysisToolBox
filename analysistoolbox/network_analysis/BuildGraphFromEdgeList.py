# Load packages
import networkx as nx
import pandas as pd

# Declare function
def BuildGraphFromEdgeList(dataframe,
                            node_1_column,
                            node_2_column,
                            is_directed=False,
                            edge_weight_column=None,
                            node_attributes_dataframe=None,
                            node_id_column=None,
                            allow_self_loops=False,
                            allow_multi_edges=False,
                            print_summary=True):
    """
    Convert a long-format edge list DataFrame into a networkx Graph/DiGraph
    (or MultiGraph/MultiDiGraph, if multi-edges are allowed).

    This is the foundational conversion function for the network_analysis module.
    Every other function in this module either accepts a networkx graph directly
    or wraps this function internally so analysts can pass a raw edge DataFrame
    without ever touching networkx themselves — the same pattern used by
    dataframe-in functions like CreateHierarchicalClusters elsewhere in this
    package.

    Teaching Note
    -------------
    Relational data — who talks to whom, which entities transact with which,
    which organizations share members — is usually collected and stored in
    "long" edge-list form: one row per relationship, with columns identifying
    the two endpoints. That tabular shape is convenient for storage and
    collection, but it is the wrong shape for the questions network analysis
    answers: who is central, which entities cluster together, how information
    or risk could propagate, which relationships are structurally critical.
    Answering those questions requires a graph object with real traversal and
    algorithmic structure, not just two columns of a DataFrame.

    Real-world edge lists are also rarely clean. The same relationship can be
    logged twice (multi-edges), an entity can be linked to itself through a
    data error or a legitimate reflexive relationship (self-loops), and
    analysts often want to reason about node-level metadata (a person's
    organization, a company's country, an account's type) that lives in a
    separate lookup table rather than in the edge list itself. Deciding how
    to handle these cases — silently, or with a transparent accounting of what
    was dropped — matters because self-loops and duplicate edges can distort
    centrality, community detection, and path-based metrics if they slip
    through unnoticed. Surfacing a drop report keeps that data-cleaning
    decision visible to the analyst instead of hidden inside a conversion
    utility.

    Parameters
    ----------
    dataframe
        A pandas DataFrame in long format, with one row per edge (relationship).
    node_1_column
        Name of the column containing the first endpoint of each edge.
    node_2_column
        Name of the column containing the second endpoint of each edge.
    is_directed
        Whether the relationship is directional (node_1 -> node_2). If True,
        a DiGraph (or MultiDiGraph) is built; otherwise an undirected Graph
        (or MultiGraph) is built. Defaults to False.
    edge_weight_column
        Optional name of a column containing edge weights. If provided, the
        values are attached to each edge as a 'weight' attribute, which is the
        attribute name networkx algorithms (e.g., shortest path, centrality)
        expect by default. Defaults to None.
    node_attributes_dataframe
        Optional DataFrame of node-level metadata (e.g., name, type,
        organization, country) to attach to nodes as attributes. Must be used
        together with node_id_column. Nodes present in this table but absent
        from the edge list are still added to the graph as isolated nodes, so
        analysts can attach a full roster of entities even if some have no
        recorded relationships yet. Defaults to None.
    node_id_column
        Name of the column in node_attributes_dataframe that identifies each
        node. Required if node_attributes_dataframe is provided. Defaults to
        None.
    allow_self_loops
        Whether to keep edges where node_1 and node_2 are the same entity. If
        False, self-loop rows are dropped before the graph is built. Defaults
        to False.
    allow_multi_edges
        Whether to keep multiple edges between the same pair of nodes. If
        True, the resulting graph is a MultiGraph/MultiDiGraph. If False,
        duplicate edges (the same unordered pair for undirected graphs, or the
        same ordered pair for directed graphs) are collapsed to a single edge,
        keeping the first occurrence. Defaults to False.
    print_summary
        Whether to print a report of how many rows were dropped for missing
        endpoints, self-loops, and duplicate/multi-edges, along with the
        resulting node and edge counts. Defaults to True.

    Returns
    -------
    networkx.Graph, networkx.DiGraph, networkx.MultiGraph, or networkx.MultiDiGraph
        The constructed graph. The graph's `.graph` dict includes a
        'edge_list_build_summary' key with the drop counts, so downstream
        functions can report on data quality without recomputing it.

    Examples
    --------
    # Build a simple undirected graph from a list of collaborations
    import pandas as pd
    edges_df = pd.DataFrame({
        'person_a': ['Alice', 'Alice', 'Bob', 'Carol'],
        'person_b': ['Bob', 'Carol', 'Carol', 'Carol'],
        'projects_together': [3, 1, 2, 4]
    })
    graph = BuildGraphFromEdgeList(
        edges_df,
        node_1_column='person_a',
        node_2_column='person_b',
        edge_weight_column='projects_together'
    )

    # Build a directed graph with node metadata attached
    node_lookup_df = pd.DataFrame({
        'entity_id': ['Alice', 'Bob', 'Carol', 'Dave'],
        'org': ['Acme', 'Acme', 'Globex', 'Globex'],
        'country': ['US', 'US', 'DE', 'DE']
    })
    directed_graph = BuildGraphFromEdgeList(
        edges_df,
        node_1_column='person_a',
        node_2_column='person_b',
        is_directed=True,
        node_attributes_dataframe=node_lookup_df,
        node_id_column='entity_id'
    )
    """
    # Validate required columns
    if node_1_column not in dataframe.columns:
        raise ValueError(f"Column '{node_1_column}' not found in dataframe.")
    if node_2_column not in dataframe.columns:
        raise ValueError(f"Column '{node_2_column}' not found in dataframe.")
    if edge_weight_column is not None and edge_weight_column not in dataframe.columns:
        raise ValueError(f"Column '{edge_weight_column}' not found in dataframe.")
    if node_attributes_dataframe is not None and node_id_column is None:
        raise ValueError("node_id_column must be provided when node_attributes_dataframe is given.")
    if node_id_column is not None:
        if node_attributes_dataframe is None:
            raise ValueError("node_attributes_dataframe must be provided when node_id_column is given.")
        if node_id_column not in node_attributes_dataframe.columns:
            raise ValueError(f"Column '{node_id_column}' not found in node_attributes_dataframe.")

    # Keep only the columns we need
    columns_to_keep = [node_1_column, node_2_column]
    if edge_weight_column is not None:
        columns_to_keep.append(edge_weight_column)
    edges = dataframe[columns_to_keep].copy()

    # Drop rows with missing endpoints -- they cannot form a valid edge
    count_before_missing_check = len(edges)
    edges = edges.dropna(subset=[node_1_column, node_2_column])
    count_dropped_for_missing_endpoints = count_before_missing_check - len(edges)

    # Identify and (optionally) drop self-loops
    is_self_loop = edges[node_1_column] == edges[node_2_column]
    count_self_loops_found = int(is_self_loop.sum())
    if not allow_self_loops:
        edges = edges[~is_self_loop]
        count_self_loops_dropped = count_self_loops_found
    else:
        count_self_loops_dropped = 0

    # Identify and (optionally) collapse multi-edges. For undirected graphs, treat
    # (a, b) and (b, a) as the same pair. Sort by str() so mixed-type node ids
    # (e.g., ints and strings) don't raise a TypeError when compared directly.
    if is_directed:
        edge_key = list(zip(edges[node_1_column], edges[node_2_column]))
    else:
        edge_key = [
            tuple(sorted((node_1, node_2), key=str))
            for node_1, node_2 in zip(edges[node_1_column], edges[node_2_column])
        ]
    edges = edges.assign(_edge_key=edge_key)

    count_multi_edges_found = int(edges.duplicated(subset='_edge_key').sum())
    if not allow_multi_edges:
        edges = edges.drop_duplicates(subset='_edge_key', keep='first')
        count_multi_edges_dropped = count_multi_edges_found
    else:
        count_multi_edges_dropped = 0
    edges = edges.drop(columns='_edge_key')

    # Choose the networkx graph class based on directedness and multi-edge support
    if allow_multi_edges:
        graph_class = nx.MultiDiGraph if is_directed else nx.MultiGraph
    else:
        graph_class = nx.DiGraph if is_directed else nx.Graph

    # Build the graph from the cleaned edge list
    if edge_weight_column is not None:
        edges = edges.rename(columns={edge_weight_column: 'weight'})
        graph = nx.from_pandas_edgelist(
            edges,
            source=node_1_column,
            target=node_2_column,
            edge_attr='weight',
            create_using=graph_class()
        )
    else:
        graph = nx.from_pandas_edgelist(
            edges,
            source=node_1_column,
            target=node_2_column,
            create_using=graph_class()
        )

    # Attach node attributes (and any isolated nodes) from the lookup table
    if node_attributes_dataframe is not None:
        attribute_columns = [
            column for column in node_attributes_dataframe.columns
            if column != node_id_column
        ]
        for _, row in node_attributes_dataframe.iterrows():
            graph.add_node(
                row[node_id_column],
                **{column: row[column] for column in attribute_columns}
            )

    # Record a build summary on the graph itself, so downstream functions can
    # surface data-quality context without recomputing it
    build_summary = {
        'is_directed': is_directed,
        'allow_self_loops': allow_self_loops,
        'allow_multi_edges': allow_multi_edges,
        'rows_dropped_for_missing_endpoints': count_dropped_for_missing_endpoints,
        'self_loops_found': count_self_loops_found,
        'self_loops_dropped': count_self_loops_dropped,
        'multi_edges_found': count_multi_edges_found,
        'multi_edges_dropped': count_multi_edges_dropped,
        'node_count': graph.number_of_nodes(),
        'edge_count': graph.number_of_edges(),
    }
    graph.graph['edge_list_build_summary'] = build_summary

    # Print a report of what was dropped, if requested
    if print_summary:
        print(f"\nBuilt {type(graph).__name__} with {build_summary['node_count']} nodes and {build_summary['edge_count']} edges.")
        if count_dropped_for_missing_endpoints:
            print(f"  Rows dropped for missing node values: {count_dropped_for_missing_endpoints}")
        print(f"  Self-loops found: {count_self_loops_found} (dropped: {count_self_loops_dropped})")
        print(f"  Duplicate/multi-edges found: {count_multi_edges_found} (dropped: {count_multi_edges_dropped})")

    # Return the graph
    return graph
