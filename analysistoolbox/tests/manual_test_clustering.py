import pandas as pd
import numpy as np
import sys

# Try importing normally, assuming -m execution or properly set PYTHONPATH
try:
    from analysistoolbox.geospatial_analysis.ConductClusterAnalysis import ConductClusterAnalysis
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"sys.path: {sys.path}")
    import os
    print(f"CWD: {os.getcwd()}")
    # Fallback: try raw path append if running as script from weird location
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from analysistoolbox.geospatial_analysis.ConductClusterAnalysis import ConductClusterAnalysis


def test_clustering():
    print("Generating dummy data...")
    data = []
    
    # NYC points (approx 10 points within <1km)
    for _ in range(10):
        data.append({'lat': 40.7128 + np.random.normal(0, 0.001), 
                     'lon': -74.0060 + np.random.normal(0, 0.001)})
        
    # LA points
    for _ in range(10):
        data.append({'lat': 34.0522 + np.random.normal(0, 0.001), 
                     'lon': -118.2437 + np.random.normal(0, 0.001)})
        
    # Noise points
    data.append({'lat': 0, 'lon': 0})
    data.append({'lat': 10, 'lon': 10})
    
    df = pd.DataFrame(data)
    
    print("Running ConductClusterAnalysis...")
    summary = ConductClusterAnalysis(
        dataframe=df,
        longitude_column='lon',
        latitude_column='lat',
        distance_km=5.0, 
        min_samples=3,
        map_clusters=False 
    )
    
    print("\nSummary Results:")
    print(summary)
    
    # Assertions
    print("\nVerifying results...")
    assert len(summary) >= 2, "Should have found at least 2 clusters (NYC, LA)"
    
    # Check if we found clusters besides noise
    non_noise = summary[summary['Cluster ID'] != -1]
    assert len(non_noise) >= 2, "Should have found NYC and LA clusters"
    
    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_clustering()
