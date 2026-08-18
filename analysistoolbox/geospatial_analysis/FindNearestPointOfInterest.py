# Load packages
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Declare function
def FindNearestPointOfInterest(dataframe,
                                points_of_interest,
                                longitude_column='lon',
                                latitude_column='lat',
                                poi_longitude_column='lon',
                                poi_latitude_column='lat',
                                id_column=None,
                                poi_id_column=None,
                                number_of_neighbors=1,
                                distance_unit='km',
                                map_matches=False):
    """
    For each point in a DataFrame of observations, find the nearest point(s)
    in a separate DataFrame of reference points of interest (POIs) -- e.g.,
    facilities, checkpoints, or prior sightings -- along with the distance
    to each.

    This is a companion to CalculateHaversineDistance: where that function
    computes every pairwise distance within a single set of points,
    FindNearestPointOfInterest answers a more targeted question that comes up
    constantly in fieldwork and OSINT analysis -- "which known location is
    each of these observations closest to, and how far away is it?" Rather
    than computing the full O(n * m) distance matrix between observations and
    reference points, it indexes the reference points in a BallTree with the
    haversine metric, so nearest-neighbor lookups scale efficiently even when
    the reference set is large (e.g., thousands of facilities or checkpoints).

    Teaching Note
    -------------
    Nearest-point analysis is one of the most common bridges between raw
    location data and an analytic judgment. A list of coordinates -- a
    sensor ping, a reported sighting, a delivery address -- means little on
    its own; it becomes useful the moment it is related to something known,
    like "2.3 km from the nearest checkpoint" or "closest to the Northgate
    warehouse." That relationship is what turns a bare coordinate into
    context an analyst can reason about: flagging observations that fall
    suspiciously far from any known facility, assigning each observation to
    its nearest service area, or measuring how access to a resource (a
    clinic, a polling place, a water source) varies across a population.

    The BallTree with a haversine metric is the standard efficient structure
    for this kind of query on the sphere. A naive approach would compute the
    distance from every observation to every reference point, which becomes
    prohibitively slow as both sets grow. A BallTree instead organizes the
    reference points hierarchically so that most candidates can be ruled out
    without ever computing their exact distance, which is why it is the same
    machinery used by ConductClusterAnalysis's DBSCAN pass.

    Parameters
    ----------
    dataframe
        A pandas DataFrame of observations, with one row per point, that you
        want to match to the nearest reference point(s).
    points_of_interest
        A pandas DataFrame of reference points (e.g., facilities,
        checkpoints, prior sightings) to search for the nearest match.
    longitude_column
        Name of the column in `dataframe` containing longitude values, in
        decimal degrees. Defaults to 'lon'.
    latitude_column
        Name of the column in `dataframe` containing latitude values, in
        decimal degrees. Defaults to 'lat'.
    poi_longitude_column
        Name of the column in `points_of_interest` containing longitude
        values, in decimal degrees. Defaults to 'lon'.
    poi_latitude_column
        Name of the column in `points_of_interest` containing latitude
        values, in decimal degrees. Defaults to 'lat'.
    id_column
        Optional name of a column in `dataframe` that uniquely identifies
        each observation. If provided, these labels are carried into the
        output. If None, the DataFrame's index is used instead. Defaults to
        None.
    poi_id_column
        Optional name of a column in `points_of_interest` that labels each
        reference point (e.g., a facility name). If provided, this label is
        returned alongside each match. If None, the `points_of_interest`
        index is used instead. Defaults to None.
    number_of_neighbors
        How many nearest reference points to return for each observation,
        ranked closest first. Defaults to 1.
    distance_unit
        Unit for the returned distances. One of 'km' (kilometers), 'mi'
        (miles), or 'nm' (nautical miles). Defaults to 'km'.
    map_matches
        Whether to generate and display an interactive Folium map showing
        each observation connected by a line to its single nearest
        reference point. Only applies when number_of_neighbors=1. Defaults
        to False.

    Returns
    -------
    pd.DataFrame
        A copy of `dataframe` with `number_of_neighbors` additional rows per
        observation (one per ranked match), plus columns for the matched
        reference point's identifier, its coordinates, the distance to it
        (named 'distance_km', 'distance_mi', or 'distance_nm' to match
        distance_unit), and 'neighbor_rank' (1 = nearest).

    Examples
    --------
    # Match field observations to the nearest known facility
    import pandas as pd
    observations_df = pd.DataFrame({
        'sighting_id': ['S1', 'S2', 'S3'],
        'lat': [34.0522, 34.1478, 33.9425],
        'lon': [-118.2437, -118.1445, -118.4081]
    })
    facilities_df = pd.DataFrame({
        'facility_name': ['Northgate Warehouse', 'Southside Depot'],
        'lat': [34.0600, 33.9500],
        'lon': [-118.2500, -118.4000]
    })

    matches_df = FindNearestPointOfInterest(
        observations_df,
        facilities_df,
        id_column='sighting_id',
        poi_id_column='facility_name'
    )
    # One row per sighting, each paired with its closest facility and the
    # distance to it in kilometers.

    # Return the three closest facilities to each sighting instead of just one
    matches_df = FindNearestPointOfInterest(
        observations_df,
        facilities_df,
        id_column='sighting_id',
        poi_id_column='facility_name',
        number_of_neighbors=3
    )
    """
    # Validate required columns
    if longitude_column not in dataframe.columns:
        raise ValueError(f"Column '{longitude_column}' not found in dataframe.")
    if latitude_column not in dataframe.columns:
        raise ValueError(f"Column '{latitude_column}' not found in dataframe.")
    if poi_longitude_column not in points_of_interest.columns:
        raise ValueError(f"Column '{poi_longitude_column}' not found in points_of_interest.")
    if poi_latitude_column not in points_of_interest.columns:
        raise ValueError(f"Column '{poi_latitude_column}' not found in points_of_interest.")
    if id_column is not None and id_column not in dataframe.columns:
        raise ValueError(f"Column '{id_column}' not found in dataframe.")
    if poi_id_column is not None and poi_id_column not in points_of_interest.columns:
        raise ValueError(f"Column '{poi_id_column}' not found in points_of_interest.")
    if number_of_neighbors < 1:
        raise ValueError("number_of_neighbors must be at least 1.")

    # Radius of the Earth in the requested unit, used to convert the
    # unitless great-circle angle returned by the BallTree into a ground
    # distance
    radius_by_unit = {
        'km': 6371.0088,
        'mi': 3958.7613,
        'nm': 3440.065,
    }
    if distance_unit not in radius_by_unit:
        raise ValueError("distance_unit must be one of 'km', 'mi', or 'nm'.")
    earth_radius = radius_by_unit[distance_unit]
    distance_column_name = f'distance_{distance_unit}'

    # Set up the observations, keeping only what we need and carrying the id
    # column (or the index) through as the point identifier
    obs_df = dataframe.copy()
    if id_column is not None:
        obs_df = obs_df[[id_column, latitude_column, longitude_column]]
        obs_df = obs_df.rename(columns={id_column: 'point_id'})
    else:
        obs_df = obs_df[[latitude_column, longitude_column]]
        obs_df['point_id'] = obs_df.index

    count_before_missing_check = len(obs_df)
    obs_df = obs_df.dropna(subset=[latitude_column, longitude_column])
    count_dropped_for_missing_coordinates = count_before_missing_check - len(obs_df)
    if count_dropped_for_missing_coordinates:
        print(f"Dropped {count_dropped_for_missing_coordinates} observation row(s) with missing coordinates.")
    obs_df = obs_df.reset_index(drop=True)

    if len(obs_df) == 0:
        raise ValueError("No observations with valid coordinates were found in dataframe.")

    # Set up the reference points of interest the same way
    poi_df = points_of_interest.copy()
    if poi_id_column is not None:
        poi_df = poi_df[[poi_id_column, poi_latitude_column, poi_longitude_column]]
        poi_df = poi_df.rename(columns={poi_id_column: 'poi_id'})
    else:
        poi_df = poi_df[[poi_latitude_column, poi_longitude_column]]
        poi_df['poi_id'] = poi_df.index

    count_before_missing_check = len(poi_df)
    poi_df = poi_df.dropna(subset=[poi_latitude_column, poi_longitude_column])
    count_dropped_for_missing_coordinates = count_before_missing_check - len(poi_df)
    if count_dropped_for_missing_coordinates:
        print(f"Dropped {count_dropped_for_missing_coordinates} point(s) of interest with missing coordinates.")
    poi_df = poi_df.reset_index(drop=True)

    if len(poi_df) == 0:
        raise ValueError("No points of interest with valid coordinates were found in points_of_interest.")

    if number_of_neighbors > len(poi_df):
        raise ValueError(
            f"number_of_neighbors ({number_of_neighbors}) cannot exceed the number of "
            f"available points of interest ({len(poi_df)})."
        )

    # Build the BallTree over the reference points using the haversine
    # metric, which expects coordinates as [lat, lon] in radians. This lets
    # us query the nearest reference point(s) for every observation without
    # computing the full observation-by-reference distance matrix.
    poi_radians = np.radians(poi_df[[poi_latitude_column, poi_longitude_column]].values)
    tree = BallTree(poi_radians, metric='haversine')

    obs_radians = np.radians(obs_df[[latitude_column, longitude_column]].values)
    central_angles, neighbor_indices = tree.query(obs_radians, k=number_of_neighbors)
    distances = central_angles * earth_radius

    # Expand into one row per (observation, ranked match) pair
    result_rows = []
    for obs_position in range(len(obs_df)):
        obs_row = obs_df.iloc[obs_position]
        for rank in range(number_of_neighbors):
            poi_row = poi_df.iloc[neighbor_indices[obs_position, rank]]
            result_rows.append({
                'point_id': obs_row['point_id'],
                latitude_column: obs_row[latitude_column],
                longitude_column: obs_row[longitude_column],
                'nearest_poi_id': poi_row['poi_id'],
                f'nearest_poi_{poi_latitude_column}': poi_row[poi_latitude_column],
                f'nearest_poi_{poi_longitude_column}': poi_row[poi_longitude_column],
                distance_column_name: distances[obs_position, rank],
                'neighbor_rank': rank + 1,
            })

    result_df = pd.DataFrame(result_rows)

    # Draw each observation connected to its single nearest reference point,
    # so mismatches (an observation oddly far from every known location)
    # stand out visually
    if map_matches:
        import folium

        if number_of_neighbors != 1:
            print("map_matches only draws the nearest (rank 1) match; additional neighbors are not plotted.")

        nearest_only = result_df[result_df['neighbor_rank'] == 1]

        center_lat = pd.concat([obs_df[latitude_column], poi_df[poi_latitude_column]]).mean()
        center_lon = pd.concat([obs_df[longitude_column], poi_df[poi_longitude_column]]).mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        for _, row in poi_df.iterrows():
            folium.Marker(
                location=[row[poi_latitude_column], row[poi_longitude_column]],
                popup=str(row['poi_id']),
                icon=folium.Icon(color='red', icon='star'),
            ).add_to(m)

        for _, row in nearest_only.iterrows():
            folium.CircleMarker(
                location=[row[latitude_column], row[longitude_column]],
                radius=5,
                color='#2563EB',
                fill=True,
                fill_color='#2563EB',
                popup=str(row['point_id']),
            ).add_to(m)
            folium.PolyLine(
                locations=[
                    [row[latitude_column], row[longitude_column]],
                    [row[f'nearest_poi_{poi_latitude_column}'], row[f'nearest_poi_{poi_longitude_column}']],
                ],
                color='lightgray',
                weight=1.5,
                opacity=0.7,
            ).add_to(m)

        try:
            from IPython.display import display
            display(m)
        except ImportError:
            pass

    return result_df
