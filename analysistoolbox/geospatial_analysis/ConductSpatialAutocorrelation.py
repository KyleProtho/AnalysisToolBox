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
    Perform Local Spatial Autocorrelation (Local Moran's I) to identify spatial clusters and outliers.

    This function calculates Local Indicators of Spatial Association (LISA) to detect local 
    hotspots (High-High), coldspots (Low-Low), and spatial outliers (High-Low or Low-High). 
    It utilizes the Local Moran's I statistic combined with a K-Nearest Neighbors (KNN) 
    spatial weights matrix. Statistical significance is determined via permutation testing, 
    allowing for the identification of areas where values are significantly more similar 
    or dissimilar to their neighbors than would be expected by chance.

    Spatial autocorrelation analysis is essential for:
      * Identifying geographic clusters of high disease incidence in epidemiology
      * Detecting localized price anomalies or supply chain disruptions in logistics
      * Pinpointing demographic shifts or localized sentiment patterns in intelligence analysis
      * Validating environmental pollution hotspots or biodiversity pockets
      * Strategic urban planning by identifying zones of extreme socioeconomic activity
      * Fraud detection by identifying geographic clusters of suspicious financial activity
      * Retail optimization by mapping high-performing store catchments and market potential

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataset containing geographic coordinates and the variable to be analyzed.
    longitude_column : str, optional
        The name of the column containing longitude values. Defaults to 'lon'.
    latitude_column : str, optional
        The name of the column containing latitude values. Defaults to 'lat'.
    value_column : str, optional
        The name of the numeric column to analyze for spatial patterns. If None, the function 
        automatically selects the first numeric column that is not the latitude or longitude 
        column. Defaults to None.
    k : int, optional
        The number of nearest neighbors to use when constructing the spatial weights matrix. 
        Defaults to 5.
    plot : bool, optional
        Whether to generate and display an interactive Folium map visualization of the 
        resulting hotspots, coldspots, and outliers. Defaults to False.

    Returns
    -------
    pd.DataFrame
        The original DataFrame augmented with 'moran_i' (local statistic), 'p_value' 
        (significance), and 'cluster_category' (e.g., 'Hotspot (High-High)', 
        'Not Significant') columns.

    Examples
    --------
    # Epidemiology: Mapping clusters of high infection rates
    import pandas as pd
    import numpy as np
    infection_data = pd.DataFrame({
        'county_id': range(100),
        'lat': [40.7 + np.random.normal(0, 0.5) for _ in range(100)],
        'lon': [-74.0 + np.random.normal(0, 0.5) for _ in range(100)],
        'infection_rate': [10 + np.random.normal(0, 5) for _ in range(100)]
    })
    
    # Introduce a simulated hotspot
    infection_data.loc[0:10, 'infection_rate'] = 50 
    
    results_df = ConductSpatialAutocorrelation(
        infection_data,
        value_column='infection_rate',
        k=8,
        plot=True
    )
    # Identifies the simulated hotspot and highlights significant clusters on a map.
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
