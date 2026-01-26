# Import packages
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def ConductClusterAnalysis(dataframe, 
                           longitude_column='lon', 
                           latitude_column='lat', 
                           distance_km=5, 
                           min_samples=5, 
                           map_clusters=False):
    """
    Perform density-based spatial clustering on geospatial data.

    This function utilizes the DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    algorithm to identify clusters of geographic points based on their spatial proximity. Unlike
    centroid-based methods like K-Means, DBSCAN can discover clusters of arbitrary shapes and
    is robust to outliers, which are labeled as 'Noise'. The function specifically employs the
    Haversine distance metric to calculate the great-circle distance between points on the 
    Earth's surface, ensuring accuracy for large-scale geospatial analysis.

    Geospatial cluster analysis is essential for:
      * Identifying periodic gathering points or high-activity zones in intelligence analysis
      * Mapping disease outbreaks or infection hotspots in epidemiology
      * Analyzing urban traffic patterns and identifying public transit bottlenecks
      * Optimizing logistics and locating optimal supply chain distribution centers
      * Detecting anomalies or "ghost" signals in sensor networks
      * Exploring environmental patterns such as deforestation or wildlife migration
      * Segmenting customer locations for localized marketing strategies

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataset containing geographic coordinates.
    longitude_column : str, optional
        The name of the column containing longitude values. Defaults to 'lon'.
    latitude_column : str, optional
        The name of the column containing latitude values. Defaults to 'lat'.
    distance_km : float, optional
        The maximum distance (epsilon) between two points to be considered neighbors, 
        measured in kilometers. Defaults to 5.
    min_samples : int, optional
        The minimum number of points required to form a dense region (cluster). 
        Defaults to 5.
    map_clusters : bool, optional
        Whether to generate and display an interactive Folium map visualization 
        of the resulting clusters. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame containing statistics for each identified cluster, including
        the cluster ID, cluster name, midpoint coordinates (mean), and the point count.

    Examples
    --------
    # Intelligence analysis: Identifying high-activity zones from signal data
    import pandas as pd
    import numpy as np
    signal_data = pd.DataFrame({
        'signal_id': range(1, 101),
        'lat': [34.05 + np.random.normal(0, 0.01) for _ in range(100)],
        'lon': [-118.24 + np.random.normal(0, 0.01) for _ in range(100)]
    })
    
    cluster_summary = ConductClusterAnalysis(
        signal_data,
        longitude_column='lon',
        latitude_column='lat',
        distance_km=0.5,
        min_samples=10,
        map_clusters=True
    )
    # Returns summary statistics and displays an interactive map of clusters
    """
    # Lazy load uncommon packages
    import folium
    from IPython.display import display
    
    # Create a copy to avoid modifying original if we aren't returning it
    df = dataframe.copy()
    
    # Drop rows with missing lat/lon
    df = df.dropna(subset=[longitude_column, latitude_column])
    
    # Coordinates in radians for Haversine distance
    coords = np.radians(df[[latitude_column, longitude_column]])
    
    # Earth radius in km
    kms_per_radian = 6371.0088
    epsilon = distance_km / kms_per_radian
    
    # Run DBSCAN
    # metric='haversine' expects [lat, lon] in radians
    db = DBSCAN(
        eps=epsilon, 
        min_samples=min_samples, 
        algorithm='ball_tree', 
        metric='haversine'
    ).fit(coords)
    df['cluster'] = db.labels_
    
    # Filter out noise (-1) for summary?? 
    # Usually noise is identified as -1. The user might want to know about noise too.
    # Bellingcat tool identifies "clusters". Noise is not a cluster.
    # I will calculate summary for all unique labels, including -1 (Noise), but maybe label it.
    
    # Summary
    summary_list = []
    unique_labels = set(db.labels_)
    
    for k in unique_labels:
        if k == -1:
            # Noise
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {k}"
            
        cluster_data = df[df['cluster'] == k]
        
        # Midpoint (mean)
        midpoint_lat = cluster_data[latitude_column].mean()
        midpoint_lon = cluster_data[longitude_column].mean()
        count = len(cluster_data)
        
        summary_list.append({
            'Cluster ID': k,
            'Cluster Name': cluster_name,
            'Midpoint Latitude': midpoint_lat,
            'Midpoint Longitude': midpoint_lon,
            'Point Count': count
        })
        
    summary_df = pd.DataFrame(summary_list)
    
    # Sort by count desc? or Cluster ID?
    summary_df = summary_df.sort_values(by='Cluster ID').reset_index(drop=True)

    if map_clusters:
        # Create map centered on average of all data (or median)
        center_lat = df[latitude_column].mean()
        center_lon = df[longitude_column].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Colors
        # distinct colors? 
        # For simplicity, use a function to generate colors or a list
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for _, row in df.iterrows():
            cluster_id = row['cluster']
            
            if cluster_id == -1:
                col = 'black'
                popup_text = "Noise"
            else:
                col = colors[cluster_id % len(colors)]
                popup_text = f"Cluster {cluster_id}"
            
            folium.CircleMarker(
                location=[row[latitude_column], row[longitude_column]],
                radius=5,
                color=col,
                fill=True,
                fill_color=col,
                popup=popup_text
            ).add_to(m)
            
        # If in a notebook, this will display.
        try:
            display(m)
        except ImportError:
            pass # Not in IPython/Jupyter

    return summary_df
