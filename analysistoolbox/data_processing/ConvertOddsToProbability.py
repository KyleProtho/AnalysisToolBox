# Load packages
import numpy as np
import pandas as pd

# Declare function
def ConvertOddsToProbability(dataframe,
                             odds_column,
                             probability_column_name=None):
    """This function converts odds to probability.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        odds_column (str): The name of the column containing the odds.
        probability_column_name (str, optional): The name of the column for the probability. Defaults to None.

    Returns:
        Pandas dataframe: Dataframe with odds converted to probability in a new column.
    """
    
    # If probability column name is not specified, set it to odds column name + "- as probability"
    if probability_column_name is None:
        probability_column_name = odds_column + " - as probability"
    
    # Convert odds to probability
    dataframe[probability_column_name] = np.where(
        (dataframe[odds_column].isnull()) | (dataframe[odds_column]+1 == 0),
        np.nan,
        dataframe[odds_column] / (1 + dataframe[odds_column])
    )
    
    # Return updated dataframe
    return dataframe

