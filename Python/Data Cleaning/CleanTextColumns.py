# Load packages
import pandas as pd

# Define function
def CleanTextColumns(dataframe):
    # Iterate over columns, and clean string-type columns
    for col_name in dataframe.columns:
        if dataframe.dtypes[col_name] == 'O':
            dataframe[col_name] = dataframe[col_name].str.strip()
            
    # Return dataframe
    return(dataframe)
