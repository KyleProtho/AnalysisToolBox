# Load packages
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Declare function
def ImputeMissingValuesUsingNearestNeighbors(dataframe,
                                             list_of_numeric_columns_to_impute,
                                             number_of_neighbors=3,
                                             averaging_method='uniform'):
    """_summary_
    This function imputes missing values in a dataframe using nearest neighbors.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        list_of_numeric_columns_to_impute (list): The list of variables to impute.
        number_of_neighbors (int, optional): The number of neighbors to use for imputation. Defaults to 3.
        averaging_method (str, optional): The weight function used in prediction. Defaults to 'uniform'.
    
    Returns:
        Pandas dataframe: An updated Pandas dataframe with imputed values.
    """
    
    # Select only the variables to impute
    dataframe_imputed = dataframe[list_of_numeric_columns_to_impute].copy()
    
    # Impute missing values using nearest neighbors
    imputer = KNNImputer(n_neighbors=number_of_neighbors,
                         weights=averaging_method)
    
    # Fit the imputer
    dataframe_imputed = imputer.fit_transform(dataframe_imputed)
    
    # Add "- Imputed" to the variable names
    list_new_column_names = []
    for variable in list_of_numeric_columns_to_impute:
        variable_imputed = variable + " - Imputed"
        list_new_column_names.append(variable_imputed)
    
    # Convert the imputed array to a dataframe
    dataframe_imputed = pd.DataFrame(dataframe_imputed,
                                     columns=list_new_column_names)
    
    # Bind the imputed dataframe to the original dataframe
    dataframe = pd.concat([dataframe, dataframe_imputed[list_new_column_names]], axis=1)
    del(dataframe_imputed)
    
    # Return the dataframe with imputed values
    return(dataframe)

# Test the function
from sklearn import datasets
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# Randomly set 10% of the values in the dataframe to NaN
iris = iris.mask(np.random.random(iris.shape) < .1)
# Impute the missing values
iris = ImputeMissingValuesUsingNearestNeighbors(dataframe=iris,
                                               list_of_numeric_columns_to_impute=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
