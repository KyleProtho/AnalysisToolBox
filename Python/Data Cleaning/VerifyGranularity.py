# Load packages
import pandas as pd

# Define function
def VerifyGranularity(dataframe,
                      list_of_key_columns,
                      set_key_as_index=True):
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
    if len(dataframe.index) != len(pd.unique(dataframe['Dataset Key'])):
        print("WARNING: Unique combination of columns you listed does not equal number of rows in dataset. Try a new combination of columns or see if dataset has duplicates.")
    
    # If Key column not being kept, drop it. Otherwise, move Key column to front of DataFrame
    if set_key_as_index != True:
        dataframe = dataframe.drop(columns=['Dataset Key'])
    else:
        dataframe.index = dataframe['Dataset Key']
        dataframe = dataframe.drop(columns=['Dataset Key'])
        
    # Return dataframe
    return(dataframe)

