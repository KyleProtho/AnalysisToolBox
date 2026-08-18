# Load packages
import numpy as np
import pandas as pd

# Declare function
def CalculateHaversineDistance(dataframe,
                                longitude_column='lon',
                                latitude_column='lat',
                                id_column=None,
                                distance_unit='km',
                                output_format='long',
                                plot_connections=False,
                                line_color='lightgray',
                                line_alpha=0.4,
                                figure_size=(10, 8)):
    """
    Calculate the great-circle (Haversine) distance between every pair of points
    in a DataFrame of latitude/longitude coordinates.

    This is the foundational distance primitive for the geospatial_analysis
    module. ConductClusterAnalysis already computes Haversine distances
    internally (via DBSCAN's 'haversine' metric) to group nearby points, but
    analysts frequently need the raw pairwise distances themselves — to rank
    which observations are closest to which, to feed a downstream network or
    routing analysis, or simply to answer "how far apart are these things?"
    without running a full clustering pass. This function exposes that
    primitive directly, computed with pure numpy so it carries no additional
    dependency.

    Teaching Note
    -------------
    Two points on a map look close or far apart based on their coordinates,
    but latitude and longitude are angular measurements on a sphere, not flat
    Cartesian coordinates — a one-degree change in longitude covers a very
    different ground distance at the equator than it does near the poles.
    Treating (lat, lon) pairs as if they were (x, y) points on a plane (e.g.,
    with ordinary Euclidean distance) silently introduces distortion that
    grows with distance from the equator and with the distance between the
    points themselves. The Haversine formula instead computes the great-circle
    distance — the shortest path along the surface of a sphere — which is why
    it is the standard building block for point-to-point distance analysis in
    GIS, logistics routing, and pattern-of-life analysis.

    Duplicate points are also worth catching before computing distances.
    OSINT and observational datasets frequently contain repeated readings of
    the same location (a sensor re-reporting, a location pinged twice), and
    two identical points always produce a distance of exactly zero. Left in
    place, they don't change any individual distance calculation, but they do
    inflate the number of pairs being computed and can clutter both the
    output table and the connecting-line visualization with redundant,
    uninformative zero-distance pairs. Deduplicating first keeps the pairwise
    result focused on genuinely distinct locations.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing point data, with one row per point.
    longitude_column
        Name of the column containing longitude values, in decimal degrees.
        Defaults to 'lon'.
    latitude_column
        Name of the column containing latitude values, in decimal degrees.
        Defaults to 'lat'.
    id_column
        Optional name of a column that uniquely identifies each point (e.g.,
        a site name or ID). If provided, these labels are used to identify
        points in the output and on the plot. If None, the DataFrame's index
        is used instead. Defaults to None.
    distance_unit
        Unit for the returned distances. One of 'km' (kilometers), 'mi'
        (miles), or 'nm' (nautical miles). Defaults to 'km'.
    output_format
        Shape of the returned DataFrame. 'long' returns one row per unique
        pair of points (point_1, point_2, distance). 'matrix' returns a
        square DataFrame of all pairwise distances, indexed and columned by
        point identifier. Defaults to 'long'.
    plot_connections
        Whether to draw a matplotlib scatter plot of the points with every
        pairwise connection drawn as a line between them. Intended for
        smaller point sets, since the number of connecting lines grows with
        the square of the number of points. Defaults to False.
    line_color
        Color of the connecting lines when plot_connections is True. Defaults
        to 'lightgray', so the lines stay faint and the points remain the
        visual focus.
    line_alpha
        Opacity of the connecting lines when plot_connections is True, from 0
        (invisible) to 1 (opaque). Defaults to 0.4.
    figure_size
        (width, height) of the plot in inches, when plot_connections is True.
        Defaults to (10, 8).

    Returns
    -------
    pd.DataFrame
        If output_format='long': one row per unique pair of points, with
        columns for the two point identifiers, their coordinates, and the
        distance between them (named 'distance_km', 'distance_mi', or
        'distance_nm' to match distance_unit).
        If output_format='matrix': a square DataFrame of pairwise distances,
        indexed and columned by point identifier.

    Examples
    --------
    # Rank how far apart a handful of field sites are from one another
    import pandas as pd
    sites_df = pd.DataFrame({
        'site_name': ['Warehouse', 'Depot A', 'Depot B', 'Depot A'],
        'lat': [34.0522, 34.1478, 33.9425, 34.1478],
        'lon': [-118.2437, -118.1445, -118.4081, -118.1445]
    })

    distances_df = CalculateHaversineDistance(
        sites_df,
        id_column='site_name',
        distance_unit='mi'
    )
    # The duplicate 'Depot A' row is dropped before distances are computed,
    # and distances_df holds one row per remaining pair, e.g. Warehouse <-> Depot B.

    # Visualize connections between the same sites
    distances_df = CalculateHaversineDistance(
        sites_df,
        id_column='site_name',
        plot_connections=True
    )
    """
    # Validate required columns
    if longitude_column not in dataframe.columns:
        raise ValueError(f"Column '{longitude_column}' not found in dataframe.")
    if latitude_column not in dataframe.columns:
        raise ValueError(f"Column '{latitude_column}' not found in dataframe.")
    if id_column is not None and id_column not in dataframe.columns:
        raise ValueError(f"Column '{id_column}' not found in dataframe.")
    if output_format not in ('long', 'matrix'):
        raise ValueError("output_format must be one of 'long' or 'matrix'.")

    # Radius of the Earth in the requested unit, used to convert the
    # unitless great-circle angle into a ground distance
    radius_by_unit = {
        'km': 6371.0088,
        'mi': 3958.7613,
        'nm': 3440.065,
    }
    if distance_unit not in radius_by_unit:
        raise ValueError("distance_unit must be one of 'km', 'mi', or 'nm'.")
    earth_radius = radius_by_unit[distance_unit]
    distance_column_name = f'distance_{distance_unit}'

    # Keep only the columns we need, and use the id column (or the index) as
    # the point identifier carried through the rest of the function
    df = dataframe.copy()
    if id_column is not None:
        df = df[[id_column, latitude_column, longitude_column]]
        df = df.rename(columns={id_column: 'point_id'})
    else:
        df = df[[latitude_column, longitude_column]]
        df['point_id'] = df.index

    # Drop rows with missing coordinates -- they cannot be located, let alone
    # measured against another point
    count_before_missing_check = len(df)
    df = df.dropna(subset=[latitude_column, longitude_column])
    count_dropped_for_missing_coordinates = count_before_missing_check - len(df)
    if count_dropped_for_missing_coordinates:
        print(f"Dropped {count_dropped_for_missing_coordinates} row(s) with missing coordinates.")

    # Ensure each point is unique before calculating distances -- duplicate
    # coordinates always produce a distance of zero and only add redundant
    # pairs to the output and the plot
    count_before_duplicate_check = len(df)
    df = df.drop_duplicates(subset=[latitude_column, longitude_column], keep='first')
    count_dropped_for_duplicate_points = count_before_duplicate_check - len(df)
    if count_dropped_for_duplicate_points:
        print(f"Dropped {count_dropped_for_duplicate_points} duplicate point(s) before calculating distances.")

    if len(df) < 2:
        raise ValueError("At least 2 unique points are required to calculate distances.")

    df = df.reset_index(drop=True)

    # Vectorized Haversine distance between every pair of points, via numpy
    # broadcasting: an (n, 1) column of coordinates against a (1, n) row of
    # the same coordinates produces the full (n, n) matrix of pairwise
    # differences in a single pass, with no explicit loop over rows
    lat_radians = np.radians(df[latitude_column].values)
    lon_radians = np.radians(df[longitude_column].values)

    delta_lat = lat_radians[:, None] - lat_radians[None, :]
    delta_lon = lon_radians[:, None] - lon_radians[None, :]

    haversine_term = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_radians[:, None]) * np.cos(lat_radians[None, :]) * np.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2 * np.arcsin(np.sqrt(np.clip(haversine_term, 0, 1)))
    distance_matrix = earth_radius * central_angle

    if output_format == 'matrix':
        result_df = pd.DataFrame(distance_matrix, index=df['point_id'], columns=df['point_id'])
        result_df.index.name = None
        result_df.columns.name = None
    else:
        # Long format: one row per unique unordered pair (i < j), so each
        # pair is reported once rather than twice (i, j) and (j, i)
        row_indices, col_indices = np.triu_indices(len(df), k=1)
        result_df = pd.DataFrame({
            'point_1': df['point_id'].values[row_indices],
            'point_2': df['point_id'].values[col_indices],
            f'{latitude_column}_1': df[latitude_column].values[row_indices],
            f'{longitude_column}_1': df[longitude_column].values[row_indices],
            f'{latitude_column}_2': df[latitude_column].values[col_indices],
            f'{longitude_column}_2': df[longitude_column].values[col_indices],
            distance_column_name: distance_matrix[row_indices, col_indices],
        })
        result_df = result_df.sort_values(by=distance_column_name).reset_index(drop=True)

    # Plot the points, connected by a faint line for every pairwise
    # combination, so the relative spread of distances is visible at a glance
    if plot_connections:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figure_size)

        row_indices, col_indices = np.triu_indices(len(df), k=1)
        for i, j in zip(row_indices, col_indices):
            ax.plot(
                [df[longitude_column].iloc[i], df[longitude_column].iloc[j]],
                [df[latitude_column].iloc[i], df[latitude_column].iloc[j]],
                color=line_color,
                alpha=line_alpha,
                linewidth=0.8,
                zorder=1,
            )

        ax.scatter(df[longitude_column], df[latitude_column], color='#2563EB', s=50, zorder=2)
        for _, row in df.iterrows():
            ax.annotate(
                str(row['point_id']),
                (row[longitude_column], row[latitude_column]),
                textcoords='offset points',
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Point-to-Point Distances')
        plt.show()

    return result_df
