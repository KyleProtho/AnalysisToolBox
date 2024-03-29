# Load packages
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd

# Declare function
def CreateBinnedColumn(dataframe,
                       numeric_column_name,
                       number_of_bins=6,
                       binning_strategy='kmeans',
                       new_column_name=None):
    """
    This function creates a binned column in a dataframe based on a numeric variable. 
    The default is to create 6 bins using the k-means binning strategy. 
    The new column name is the name of the variable to bin with '- Binned' appended to it.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        numeric_column_name (str): The name of the variable to bin. Should be a numeric variable.
        number_of_bins (int, optional): The number of bins to create. Defaults to 6.
        binning_strategy (str, optional): The strategy to use to create the bins. Defaults to 'kmeans'.
        new_column_name (str, optional): The name of the new column containing the bins. Defaults to None.

    Returns:
        Pandas dataframe: An updated Pandas dataframe with the bins for the variable to bin
    """
    
    # Create new column name if not provided
    if new_column_name is None:
        new_column_name = numeric_column_name + '- Binned'
        
    # Select only the variable to bin
    dataframe_bin = dataframe[[numeric_column_name]].copy()
    
    # Keep complete cases only
    dataframe_bin = dataframe_bin.dropna()
    
    # Keep only finite values
    dataframe_bin = dataframe_bin[np.isfinite(dataframe_bin).all(1)]
        
    # Create the array to bin
    array_to_bin = np.array(dataframe_bin[numeric_column_name]).reshape(-1, 1)
    
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

