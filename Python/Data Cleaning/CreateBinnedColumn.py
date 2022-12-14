from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd

def CreateBinnedColumn(dataframe,
                       variable_to_bin,
                       number_of_bins=6,
                       binning_strategy='kmeans',
                       new_column_name=None):
    # Create new column name if not provided
    if new_column_name is None:
        new_column_name = variable_to_bin + '- Binned'
        
    # Create the array to bin
    array_to_bin = np.array(dataframe[variable_to_bin]).reshape(-1, 1)
    
    # Create the discretizer
    binner = KBinsDiscretizer(n_bins=number_of_bins, 
                              strategy=binning_strategy,
                              encode='ordinal')
    
    # Fit the discretizer
    binner.fit(array_to_bin)
    
    # Create the binned variable
    dataframe[new_column_name] = binner.transform(array_to_bin)
    
    # Return the dataframe
    return dataframe

# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris = CreateBinnedColumn(
#     dataframe=iris, 
#     variable_to_bin='sepal length (cm)'
# )
