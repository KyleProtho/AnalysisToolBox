import numpy as np
import pandas as pd

def CreateRareCategoryColumn(dataframe,
                             categorical_variable,
                             rare_category_label="Other",
                             rare_category_threshold=0.05,
                             new_column_suffix=None):
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

