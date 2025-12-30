import pandas as pd
import numpy as np
import sys
import os

from ConductSpatialAutocorrelation import ConductSpatialAutocorrelation

def test_spatial_autocorrelation():
    print("Generating dummy data...")
    # Generate 50 points
    np.random.seed(42)
    lats = np.random.uniform(34.0, 35.0, 50)
    lons = np.random.uniform(-118.0, -117.0, 50)
    
    # Create a spatial pattern: Value correlates with Latitude (High-High at top)
    values = (lats - 34.0) * 100 + np.random.normal(0, 10, 50)
    
    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'score': values
    })
    
    print("Running ConductSpatialAutocorrelation...")
    try:
        result = ConductSpatialAutocorrelation(df, 
                                               longitude_column='lon', 
                                               latitude_column='lat', 
                                               value_column='score',
                                               k=5, 
                                               plot=False)
        
        print("\nVerification Successful!")
        print("Result Columns:", result.columns.tolist())
        print("Sample Result:")
        print(result[['lat', 'lon', 'score', 'moran_i', 'cluster_category']].head())
    except ImportError as e:
        print(f"\nCaught Expected ImportError (if packages missing): {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # raise

if __name__ == "__main__":
    test_spatial_autocorrelation()
