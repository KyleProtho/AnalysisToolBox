# Load packages
import pandas as pd
import numpy as np

# Declare function
def AddTPeriodColumn(dataframe,
                     date_column,
                     t_period_interval="days",
                     t_period_column_name=None):
    """
    This function adds a T-period column to a dataframe. 
    The T-period column is the number of intervals (e.g., days or weeks) since the earliest date in the dataframe.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        date_column (str): The name of the column containing the date. Options are "days", "weeks", "months", or "years"
        t_period_interval (str, optional): The interval or time object to use for the T-period column. Defaults to "days".
        t_period_column_name (str, optional): The name of the T-period column. Defaults to None. If None, the column name is "T Period in " + t_period_interval.

    Returns:
        Pandas dataframe: An updated Pandas dataframe with a T-period column.
    """
    
    # Ensure that column is a date datatype
    if dataframe[date_column].dtypes != "<M8[ns]":
        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
        
    # Set T-period column name
    if t_period_column_name == None:
        t_period_column_name = "T Period in " + t_period_interval

    # Calculate difference from earliest date in interval specified
    earliest_time = min(dataframe[date_column])
    dataframe[t_period_column_name] = dataframe[date_column] - earliest_time
    if t_period_interval == "days":
        dataframe[t_period_column_name] = dataframe[t_period_column_name].dt.days
    if t_period_interval == "weeks":
        dataframe[t_period_column_name] = dataframe[t_period_column_name] / np.timedelta64(1, 'W')
        dataframe[t_period_column_name] = dataframe[t_period_column_name].apply(np.floor)
    if t_period_interval == "months":
        dataframe[t_period_column_name] = dataframe[t_period_column_name] / np.timedelta64(1, 'M')
        dataframe[t_period_column_name] = dataframe[t_period_column_name].apply(np.floor)
    if t_period_interval == "years":
        dataframe[t_period_column_name] = dataframe[t_period_column_name] / np.timedelta64(1, 'Y')
        dataframe[t_period_column_name] = dataframe[t_period_column_name].round(0)
    
    # Return updated dataframe
    return(dataframe)
