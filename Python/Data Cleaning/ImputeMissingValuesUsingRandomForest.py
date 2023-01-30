import pandas as pd
from missingpy import MissForest

def ImputeMissingValuesUsingRandomForest(dataframe,
                                         number_of_trees=100,
                                         maximum_depth=None,
                                         random_seed=412):
    """_summary_
    This function imputes missing values in a dataframe using random forest imputation.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        number_of_trees (int, optional): The number of trees in the forest. Defaults to 100.
        maximum_depth (int, optional): The maximum depth of each tree. Defaults to None.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 412.
        
    Returns:
        Pandas dataframe: An updated Pandas dataframe with imputed values.
    """
    # Select only the variables to impute
    dataframe_imputed = dataframe.copy()
    
    # Impute missing values using nearest neighbors
    imputer = MissForest(n_estimators=number_of_trees,
                         random_state=random_seed,
                         max_depth=maximum_depth)
    
    # Fit the imputer
    dataframe_imputed = imputer.fit_transform(dataframe_imputed)
    
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