# Load packages
import pandas as pd
from sklearn.impute import KNNImputer

def ImputeMissingValuesUsingNearestNeighbors(dataframe,
                                             list_of_variables,
                                             number_of_neighbors=3,
                                             averaging_method='uniform'):
    # Select only the variables to impute
    dataframe_imputed = dataframe[list_of_variables].copy()
    dataframe_imputed = dataframe_imputed.reset_index()
    dataframe_imputed = dataframe_imputed[list_of_variables]
    
    # Impute missing values using nearest neighbors
    imputer = KNNImputer(n_neighbors=number_of_neighbors,
                         weights=averaging_method)
    
    # Fit the imputer
    dataframe_imputed = imputer.fit_transform(dataframe)
    
    # Add "- Imputed" to the variable names
    list_new_column_names = []
    for variable in list_of_variables:
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