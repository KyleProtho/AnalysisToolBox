import pandas as pd
import numpy as np

def AddLeadingZeros(dataframe,
                    column_name,
                    fixed_length=None,
                    add_as_new_column=False):
    """_summary_
    This function adds leading zeros to a column. If fixed_length is not specified, the longest string in the column is used as the fixed length. If add_as_new_column is set to True, the new column is added to the dataframe. Otherwise, the original column is updated.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        column_name (str): Name of column to add leading zeros to
        fixed_length (int, optional): The length each value in the column should be. Defaults to None.
        add_as_new_column (bool, optional): Whether the updated values with leading zeros should be added as a new column. Defaults to False.
    
    Returns:
        Pandas dataframe: An updated Pandas dataframe with leading zeros added to the specified column.
    """
    
    # If fixed length not specified, set the longest string as the fixed length
    if fixed_length == None:
        fixed_length = max(dataframe[column_name].astype(str).str.len())
    
    # If adding as new column, change the column name
    if add_as_new_column:
        new_column_name = column_name + ' - with leading 0s'
        dataframe[new_column_name] = np.where(
            dataframe[column_name].isna(),
            np.nan,
            dataframe[column_name].astype(str).str.zfill(fixed_length)
        )
        dataframe[new_column_name] = dataframe[new_column_name].str.replace(".0", "",
                                                                            regex=False)
    else:
        dataframe[column_name] = np.where(
            dataframe[column_name].isna(),
            np.nan,
            dataframe[column_name].astype(str).str.zfill(fixed_length)
        )
        dataframe[column_name] = dataframe[column_name].str.replace(".0", "",
                                                                    regex=False)
    
    # Return updated dataframe
    return(dataframe)