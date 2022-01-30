# Load packages
import pandas as pd
import numpy as np

# Define AddTPeriodColumn function
def AddTPeriodColumn(dataframe,
                     date_column_name,
                     t_period_interval="days",
                     t_period_column_name=None):
    # Ensure that column is a date datatype
    if dataframe[date_column_name].dtypes != "<M8[ns]":
        dataframe[date_column_name] = pd.to_datetime(dataframe[date_column_name])
        
    # Set T-period column name
    if t_period_column_name == None:
        t_period_column_name = "T Period in " + t_period_interval

    # Calculate difference from earliest date in interval specified
    earliest_time = min(dataframe[date_column_name])
    dataframe[t_period_column_name] = dataframe[date_column_name] - earliest_time
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
        dataframe[t_period_column_name] = dataframe[t_period_column_name].apply(np.floor)
    
    # Return updated dataframe
    return(dataframe)



# # Import test data
# df_products = pd.read_excel("C:/Users/oneno/OneDrive/Data/Risk Modeling - Practice Datasets/Predicting Count of Products Sold.xlsx",
#                             sheet_name="Train Final")
# # Test function
# df_products = AddTPeriodColumn(dataframe=df_products,
#                                date_column_name="Date")
# df_products = AddTPeriodColumn(dataframe=df_products,
#                                date_column_name="Date",
#                                t_period_interval='months')

