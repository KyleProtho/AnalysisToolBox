# Load packages
import pandas as pd
import numpy as np

# Declare function
def AddTPeriodColumn(dataframe,
                     date_column_name,
                     t_period_interval="days",
                     t_period_column_name=None):
    """
    Calculate elapsed time periods from the earliest date in a DataFrame.

    This function adds a column that measures the number of time intervals that have elapsed
    since the earliest date in the dataset. Each row receives a T-period value representing
    how many intervals (days, weeks, months, or years) have passed between the earliest date
    and that row's date. This creates a normalized time index starting from 0 (the earliest
    date) that is essential for time series analysis and longitudinal studies.

    The function is particularly useful for:
      * Cohort analysis and retention studies
      * Time series modeling and forecasting
      * Tracking progression over time from a baseline
      * Normalizing dates across different starting points
      * Panel data analysis with time-based indexing
      * Event study analysis in finance and economics
      * Customer lifetime value calculations

    The T-period value is always 0 for the row(s) with the earliest date and increases
    for subsequent dates. The function automatically converts the date column to datetime
    format if needed and floors values for weeks, months, and years to ensure integer counts.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing at least one column with date or datetime values.
        The DataFrame will be modified by adding the T-period column.
    date_column_name
        Name of the column containing date values from which to calculate time periods.
        The column will be converted to datetime format if not already in that format.
    t_period_interval
        Unit of time for measuring elapsed periods. Must be one of: 'days', 'weeks',
        'months', or 'years'. The T-period column will count the number of these units
        since the earliest date. Defaults to 'days'.
    t_period_column_name
        Custom name for the new T-period column. If None, the column will be automatically
        named 'T Period in {interval}' (e.g., 'T Period in days'). Defaults to None.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column containing the T-period values.
        Each value represents the number of complete intervals since the earliest date
        in the dataset. The earliest date(s) will have a T-period value of 0.

    Examples
    --------
    # Calculate days since first event for user activity data
    import pandas as pd
    activity = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2],
        'event_date': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-01', '2023-01-08']
    })
    activity = AddTPeriodColumn(activity, 'event_date', t_period_interval='days')
    # Adds 'T Period in days': [0, 4, 9, 0, 7] - days since earliest date (2023-01-01)

    # Track customer cohorts by weeks since first purchase
    purchases = pd.DataFrame({
        'customer': ['A', 'A', 'B', 'B', 'A'],
        'purchase_date': pd.date_range('2023-01-01', periods=5, freq='5D')
    })
    purchases = AddTPeriodColumn(
        purchases,
        'purchase_date',
        t_period_interval='weeks',
        t_period_column_name='Week Number'
    )
    # Custom column name 'Week Number' with weekly intervals

    # Measure months elapsed in a longitudinal study
    patient_visits = pd.DataFrame({
        'patient_id': [101, 101, 101, 102, 102],
        'visit_date': ['2023-01-15', '2023-03-22', '2023-06-10', '2023-01-15', '2023-05-01']
    })
    patient_visits = AddTPeriodColumn(
        patient_visits,
        'visit_date',
        t_period_interval='months'
    )
    # Adds 'T Period in months': [0, 2, 4, 0, 3] - months since baseline

    # Track yearly progression for financial data
    financials = pd.DataFrame({
        'company': ['ACME', 'ACME', 'ACME'],
        'fiscal_year_end': ['2020-12-31', '2021-12-31', '2022-12-31']
    })
    financials = AddTPeriodColumn(
        financials,
        'fiscal_year_end',
        t_period_interval='years',
        t_period_column_name='Years Since Baseline'
    )
    # Adds 'Years Since Baseline': [0, 1, 2]

    """
    
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
        dataframe[t_period_column_name] = dataframe[t_period_column_name].round(0)
    
    # Return updated dataframe
    return(dataframe)
