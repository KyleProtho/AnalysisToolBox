# Load packages
import numpy as np
import pandas as pd

# Declare function
def ConvertOddsToProbability(dataframe,
                             odds_columns,
                             probability_column_name=None):
    
    # If probability column name is not specified, set it to odds column name + "- as probability"
    if probability_column_name is None:
        probability_column_name = odds_columns + " - as probability"
    
    # Convert odds to probability
    dataframe[probability_column_name] = np.where(
        (dataframe[odds_columns].isnull()) | (dataframe[odds_columns]+1 == 0),
        np.nan,
        dataframe[odds_columns] / (1 + dataframe[odds_columns])
    )
    
    # Return updated dataframe
    return dataframe

