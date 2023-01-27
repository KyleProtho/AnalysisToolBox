import numpy as np
import pandas as pd

def CreateRareCategoryColumn(dataframe,
                             categorical_variable,
                             rare_category_label="Other",
                             rare_category_threshold=0.05,
                             new_column_suffix=None):
    """_summary_
    This function creates a new column in a dataframe with rare categories. The default is to label rare categories as "Other". The default threshold is 5%. If new_column_suffix is not specified, the new column name is the original column name with " (with Other)" appended to it.
    
    Args:
        dataframe (_type_): Pandas dataframe
        categorical_variable (str): The name of the column containing the categorical variable.
        rare_category_label (str, optional): The label to use for rare categories. Defaults to "Other".
        rare_category_threshold (float, optional): The relative frequency threshold for rare categories. Defaults to 0.05.
        new_column_suffix (str, optional): The suffix to append to the original column name to create the new column name. Defaults to None.

    Returns:
        _type_: An updated Pandas dataframe with the new column containing rare categories
    """
    # If new column name is not provided, create one
    if new_column_suffix is None:
        new_column_suffix = " (with " + rare_category_label + ")"
    
    # Get relative frequency of each category
    data_value_relative_frequency = dataframe[categorical_variable].value_counts(normalize=True)
    data_value_relative_frequency = pd.DataFrame(data_value_relative_frequency)
    
    # Add flag for rare categories
    data_value_relative_frequency['rare'] = data_value_relative_frequency[categorical_variable] < rare_category_threshold
    
    # Create new column name
    new_column_name = categorical_variable + " " + new_column_suffix
    
    # Get list of rare categories
    rare_values = data_value_relative_frequency[data_value_relative_frequency['rare'] == True].index.tolist()
    
    # Create new column in original dataframe with rare categories
    dataframe[new_column_name] = np.where(
        dataframe[categorical_variable].isnull(), np.nan,
        np.where(dataframe[categorical_variable].isin(rare_values), rare_category_label, 
        dataframe[categorical_variable])
    )
    
    # Return the dataframe
    return dataframe

