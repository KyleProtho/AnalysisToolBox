# Load packages
import pandas as pd

# Define function
def AddDateNumberColumns(dataframe,
                         date_column_name):
    
    # Extract year from date
    new_column_name = date_column_name + '.Year'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).year

    # Extract quarter from date
    new_column_name = date_column_name + '.Quarter'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).quarter

    # Extract month number from date
    new_column_name = date_column_name + '.Month'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).month

    # Extract week number from date
    new_column_name = date_column_name + '.Week'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).week

    # Extract day from date
    new_column_name = date_column_name + '.Day'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).day
    
    # Extract day of the week
    new_column_name = date_column_name + '.DayOfWeek'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).dayofweek
    print("Note: .DayOfWeek is 0-based starting on Monday (i.e. 0 = Monday, 6 = Sunday).")
    
    # Return dataframe
    return(dataframe)
