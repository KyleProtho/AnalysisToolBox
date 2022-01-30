# Load packages
import pandas as pd

# Define AddTPeriodColumn function
def AddTPeriodColumn(dataframe,
                     date_column_name,
                     t_period_interval="days",
                     t_period_column_name=None):
    # Ensure that column is a date datatype
    if dataframe[date_column_name].dtypes != "<M8[ns]":
        dataframe[date_column_name] = pd.to_datetime(dataframe[date_column_name])

    # Calculate difference from earliest date in interval specified
    earliest_time = min(dataframe[date_column_name])
    if t_period_column_name == None:
        t_period_column_name = "T Period in " + t_period_interval
    dataframe[t_period_column_name] = dataframe[date_column_name] - earliest_time
    if t_period_interval == "days":
        dataframe[t_period_column_name] = dataframe[t_period_column_name].dt.days
        
    # Return updated dataframe
    return(dataframe)



# Import test data
df_products = pd.read_excel("C:/Users/oneno/OneDrive/Data/Risk Modeling - Practice Datasets/Predicting Count of Products Sold.xlsx",
                            sheet_name="Train Final")
# Test function
df_products = AddTPeriodColumn(dataframe=df_products,
                               date_column_name="Date")
