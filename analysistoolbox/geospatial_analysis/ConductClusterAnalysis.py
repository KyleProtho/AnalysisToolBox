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
    Conducts cluster analysis on geospatial data using the DBSCAN algorithm.
     Mimics the functionality of Bellingcat's geoclustering tool.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing geospatial data.
        longitude_column (str): Name of the column containing longitude values. Defaults to 'lon'.
        latitude_column (str): Name of the column containing latitude values. Defaults to 'lat'.
        distance_km (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.5 km.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. Defaults to 5.
        map_clusters (bool): If True, displays a folium map of the clusters. Defaults to False.

    Returns:
        pd.DataFrame: A summary dataframe containing cluster statistics (midpoint, count).
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
