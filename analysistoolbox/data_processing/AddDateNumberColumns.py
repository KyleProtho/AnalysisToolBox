# Load packages
import pandas as pd

# Define function
def AddDateNumberColumns(dataframe,
                         date_column):
    """
    This function adds year, quarter, month, and week number columns to a dataset based on a date column.
    
    Args:
    dataframe (Pandas dataframe): Pandas dataframe
    date_column (str): Name of column containing dates

    Returns:
    Pandas dataframe: An updated Pandas dataframe with date number columns.
    """
    
    # Extract year from date
    new_column_name = date_column + '.Year'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column]).year

    # Extract quarter from date
    new_column_name = date_column + '.Quarter'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column]).quarter

    # Extract month number from date
    new_column_name = date_column + '.Month'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column]).month

    # Extract week number from date
    new_column_name = date_column + '.Week'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column]).week

    # Extract day from date
    new_column_name = date_column + '.Day'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column]).day
    
    # Extract day of the week
    new_column_name = date_column + '.DayOfWeek'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column]).dayofweek
    print("Note: .DayOfWeek is 0-based starting on Monday (i.e. 0 = Monday, 6 = Sunday).")
    
    # Return dataframe
    return(dataframe)

