# Load packages
import pandas as pd

# Define function
def CleanTextColumns(dataframe):
    """_summary_
    This function cleans string-type columns in a dataframe by removing leading and trailing spaces.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        
    Returns:
        Pandas dataframe: An updated Pandas dataframe with cleaned string-type columns.
    """
    
    # Iterate over columns, and clean string-type columns
    for col_name in dataframe.columns:
        if dataframe.dtypes[col_name] == 'O':
            dataframe[col_name] = dataframe[col_name].str.strip()
            
    # Return dataframe
    return(dataframe)
