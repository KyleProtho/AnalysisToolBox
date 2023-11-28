# Load packages 
import numpy as np
import pandas as pd

# Declare function
def CreateRareCategoryColumn(dataframe,
                             categorical_column_name,
                             rare_category_label="Other",
                             rare_category_threshold=0.01,
                             new_column_suffix=None):
    """
    This function creates a new column in a dataframe with rare categories. 
    The default is to label rare categories as "Other". 
    The default threshold is 1%. If new_column_suffix is not specified, the new column name is the original column name with " (with Other)" appended to it.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        categorical_column_name (str): The name of the column containing the categorical variable.
        rare_category_label (str, optional): The label to use for rare categories. Defaults to "Other".
        rare_category_threshold (float, optional): The relative frequency threshold for rare categories. Defaults to 0.01.
        new_column_suffix (str, optional): The suffix to append to the original column name to create the new column name. Defaults to None.

    Returns:
        Pandas dataframe: An updated Pandas dataframe with the new column containing rare categories
    """
    
    # If new column name is not provided, create one
    if new_column_suffix is None:
        new_column_suffix = " (with " + rare_category_label + ")"
    
    # Get relative frequency of each category
    data_value_relative_frequency = dataframe[categorical_column_name].value_counts(normalize=True)
    data_value_relative_frequency = pd.DataFrame(data_value_relative_frequency)
    
    # Add flag for rare categories
    data_value_relative_frequency['rare'] = data_value_relative_frequency[categorical_column_name] < rare_category_threshold
    
    # Create new column name
    new_column_name = categorical_column_name + " " + new_column_suffix
    
    # Get list of rare categories
    rare_values = data_value_relative_frequency[data_value_relative_frequency['rare'] == True].index.tolist()
    
    # Create new column in original dataframe with rare categories
    dataframe[new_column_name] = np.where(
        dataframe[categorical_column_name].isnull(), np.nan,
        np.where(dataframe[categorical_column_name].isin(rare_values), rare_category_label, 
        dataframe[categorical_column_name])
    )
    
    # Return the dataframe
    return dataframe

