# Load packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Declare function
def ImputeMissingValuesUsingRandomForest(dataframe,
                                         list_of_columns_to_impute=None,
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
    # Create a copy of the dataframe
    dataframe_imputed = dataframe.copy()
    
    # Identify the columns with missing values, if not specified
    if list_of_columns_to_impute==None:
        list_of_columns_to_impute = dataframe_imputed.columns[dataframe_imputed.isnull().sum() > 0]
    
    # Create a random forest regressor.
    imputer = RandomForestRegressor(
        n_estimators=number_of_trees,
        max_depth=maximum_depth,
        random_state=random_seed
    )
    
    # Iterate over the columns with missing values.
    for column in list_of_columns_to_impute:
        # Filter to only rows with non-missing values in the column.
        df_temp = dataframe_imputed[dataframe_imputed[column].notnull()].copy()
        
        # Identify the columns that do not have missing values
        list_eligible_predictor_columns = df_temp.columns[df_temp.isnull().sum() == 0]
        
        # Remove the column with missing values from the list of eligible predictor columns
        list_eligible_predictor_columns = list_eligible_predictor_columns.drop(column)
        
        # Split the data into training and testing sets.
        X_train = df_temp[list_eligible_predictor_columns]
        y_train = df_temp[column]
        X_test = dataframe_imputed[dataframe_imputed[column].isna()][list_eligible_predictor_columns]

        # Train the random forest regressor on the training data.
        imputer.fit(X_train, y_train)
        
        # Impute the missing values in the original dataframe.
        dataframe_imputed.loc[dataframe_imputed[column].isnull(), column] = imputer.predict(X_test)
    
    # Add "- Imputed" to the variable names
    list_new_column_names = []
    for variable in list_of_columns_to_impute:
        variable_imputed = variable + " - Imputed"
        list_new_column_names.append(variable_imputed)
    
    # Convert the imputed array to a dataframe
    dataframe_imputed = pd.DataFrame(dataframe_imputed,
                                     columns=list_new_column_names)
    
    # Bind the imputed dataframe to the original dataframe
    dataframe = pd.concat([dataframe, dataframe_imputed[list_new_column_names]], axis=1)
    
    # Return the dataframe with imputed values
    return(dataframe)


# Test the function
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, 6]
})

imputed_data = ImputeMissingValuesUsingRandomForest(data)

imputed_data = ImputeMissingValuesUsingRandomForest(data, number_of_trees=200)

imputed_data = ImputeMissingValuesUsingRandomForest(data, maximum_depth=5)

imputed_data1 = ImputeMissingValuesUsingRandomForest(data, random_seed=123)
imputed_data2 = ImputeMissingValuesUsingRandomForest(data, random_seed=123)
pd.testing.assert_frame_equal(imputed_data1, imputed_data2)
