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
        
    # Select only the variable to bin
    dataframe_bin = dataframe[[variable_to_bin]].copy()
    
    # Keep complete cases only
    dataframe_bin = dataframe_bin.dropna()
    
    # Keep only finite values
    dataframe_bin = dataframe_bin[np.isfinite(dataframe_bin).all(1)]
        
    # Create the array to bin
    array_to_bin = np.array(dataframe_bin[variable_to_bin]).reshape(-1, 1)
    
    # Create the discretizer
    binner = KBinsDiscretizer(n_bins=number_of_bins, 
                              strategy=binning_strategy,
                              encode='ordinal')
    
    # Fit the discretizer
    binner.fit(array_to_bin)
    
    # Create the binned variable
    dataframe_bin[new_column_name] = binner.transform(array_to_bin)
    
    # Merge the binned variable back into the original dataframe
    dataframe = dataframe.merge(
        dataframe_bin[[new_column_name]],
        how='left',
        right_index=True,
        left_index=True
    )
    
    # Return the dataframe
    return dataframe

# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris = CreateBinnedColumn(
#     dataframe=iris, 
#     variable_to_bin='sepal length (cm)'
# )
