# Load packages
import pandas as pd

# Define function
def VerifyGranularity(dataframe,
                      list_of_key_columns,
                      keep_key_column=True):
    # Create key column from list of key columns
    first_col = list_of_key_columns[0]
    dataframe['Dataset Key'] = dataframe[first_col].astype(str)
    
    # If more than 1 column listed, concatenate columns together
    if len(list_of_key_columns) > 1:
        for key_col in list_of_key_columns[1:]:
            dataframe['Dataset Key'] = dataframe['Dataset Key'].astype(str) + " -- " + dataframe[key_col].astype(str)
            
    # Print row count and distinct key of Dataset Key
    print("Number of rows in dataframe:", str(len(dataframe.index)))
    print("Distinct count of Dataset Keys:", str(len(pd.unique(dataframe['Dataset Key']))))
    
    # If Key column not being kept, drop it. Otherwise, move Key column to front of DataFrame
    if keep_key_column != True:
        dataframe = dataframe.drop(columns=['Dataset Key'])
    else:
        first_column = dataframe.pop('Dataset Key')
        dataframe.insert(0, 'Dataset Key', first_column)
        
    # Return dataframe
    return(dataframe)