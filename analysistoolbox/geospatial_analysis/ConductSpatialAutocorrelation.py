import pandas as pd
import numpy as np

# Declare function
def ConductSpatialAutocorrelation(dataframe, 
                                  longitude_column='lon', 
                                  latitude_column='lat', 
                                  value_column=None,
                                  k=5, 
                                  plot=False):
    """
    Conducts Local Spatial Autocorrelation (Local Moran's I) analysis on a dataframe of points.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe containing geospatial data.
        longitude_column (str): Name of the column containing longitude values. Defaults to 'lon'.
        latitude_column (str): Name of the column containing latitude values. Defaults to 'lat'.
        value_column (str, optional): Name of the column containing the value to analyze. 
                                      If None, uses the first numeric column that is not lat/lon.
        k (int): Number of nearest neighbors to use for spatial weights. Defaults to 5.
        plot (bool): If True, displays a folium map of the results. Defaults to False.
        
    Returns:
        pd.DataFrame: DataFrame with original data plus 'moran_i', 'p_value', 'cluster_category' columns.
    """
    # Lazy load dependencies
    try:
        from esda.moran import Moran_Local
        from libpysal.weights import KNN
        import folium
        from IPython.display import display
        import matplotlib.colors as mcolors
    except ImportError as e:
        raise ImportError("This function requires 'esda', 'libpysal', and 'folium'. Please install them to use this feature.") from e

    # Validation
    df = dataframe.copy()
    if longitude_column not in df.columns or latitude_column not in df.columns:
        raise ValueError(f"Columns {longitude_column} and {latitude_column} must exist in the dataframe.")
    
    # Determine value column if not provided
    if value_column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in [longitude_column, latitude_column]]
        if not numeric_cols:
            raise ValueError("No suitable numeric column found for analysis. Please specify 'value_column'.")
        value_column = numeric_cols[0]
        print(f"No value_column specified. defaulting to '{value_column}'")
    
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in dataframe.")

    # Drop NaNs
    original_len = len(df)
    df = df.dropna(subset=[longitude_column, latitude_column, value_column])
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} rows with missing values.")

    if len(df) < k + 1:
        raise ValueError(f"Not enough data points ({len(df)}) for k={k} nearest neighbors.")

    # Prepare coordinates and weights
    # KNN expects (n, 2) array. libpysal uses Euclidian distance by default for KNN if passed arrays.
    # For lat/lon, strictly speaking we might want arc distance but for KNN it's usually fine as approximation 
    # if area is not global. 
    coords = df[[longitude_column, latitude_column]].values
    w = KNN.from_array(coords, k=k)
    w.transform = 'r' # Row-standardize weights

    # Conduct Local Moran's I
    y = df[value_column].values
    moran_local = Moran_Local(y, w)

    # Assign results back to DataFrame
    df['moran_i'] = moran_local.Is
    df['p_value'] = moran_local.p_sim
    
    # Classify Quadrants
    # 1: HH, 2: LH, 3: LL, 4: HL (Standard esda quadrant labels)
    # Check significance
    sig = moran_local.p_sim < 0.05
    
    # Create interpretable labels
    # q values: 1 HH, 2 LH, 3 LL, 4 HL
    # We want to label them only if significant.
    
    def get_label(row_idx, q_val, p_val):
        if p_val >= 0.05:
            return "Not Significant"
        if q_val == 1:
            return "Hotspot (High-High)"
        elif q_val == 3:
            return "Coldspot (Low-Low)"
        elif q_val == 2:
            return "Spatial Outlier (Low-High)"
        elif q_val == 4:
            return "Spatial Outlier (High-Low)"
        else:
            return "Unknown" # Should not happen

    labels = [get_label(i, q, p) for i, (q, p) in enumerate(zip(moran_local.q, moran_local.p_sim))]
    df['cluster_category'] = labels
    
    # Print Summary
    print("-" * 30)
    print(f"Local Spatial Autocorrelation Analysis Results")
    print(f"Variable: {value_column}")
    print("-" * 30)
    counts = df['cluster_category'].value_counts()
    for category, count in counts.items():
        print(f"{category}: {count}")
    print("-" * 30)

    # Plotting
    if plot:
        center_lat = df[latitude_column].mean()
        center_lon = df[longitude_column].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Color mapping
        # HH: Red, LL: Blue, Outliers: Orange/LightBlue, NS: Gray
        color_map = {
            "Hotspot (High-High)": "red",
            "Coldspot (Low-Low)": "blue",
            "Spatial Outlier (Low-High)": "lightblue",
            "Spatial Outlier (High-Low)": "orange",
            "Not Significant": "gray"
        }

        for idx, row in df.iterrows():
            cat = row['cluster_category']
            col = color_map.get(cat, 'gray')
            
            # Popup info
            popup_text = f"""
            <b>Index:</b> {idx}<br>
            <b>Value:</b> {row[value_column]:.2f}<br>
            <b>Category:</b> {cat}<br>
            <b>Moran I:</b> {row['moran_i']:.3f}
            """
            
            # Simple CircleMarker
            folium.CircleMarker(
                location=[row[latitude_column], row[longitude_column]],
                radius=5,
                color=col,
                fill=True,
                fill_color=col,
                fill_opacity=0.7 if cat != "Not Significant" else 0.2,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)
        
        # Display map
        try:
            display(m)
        except:
            pass # Non-notebook environment
            
    return df
