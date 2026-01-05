# Load packages
import pandas as pd

# Define function
def AddDateNumberColumns(dataframe,
                         date_column_name):
    """
    Add temporal components (year, quarter, month, day, day of week) as separate columns to a DataFrame.

    This function enriches a pandas DataFrame by extracting and adding multiple date components
    as new columns based on an existing date column. It creates five new columns with standardized
    naming conventions, making it easy to perform temporal analysis, grouping, and filtering.

    The function is particularly useful for:
      * Time series analysis and seasonal pattern detection
      * Grouping data by temporal periods (yearly, quarterly, monthly)
      * Creating date-based filters and segments
      * Generating time-based reports and visualizations
      * Preparing data for machine learning models with temporal features
      * Calendar-based business intelligence and analytics

    Each new column is automatically named using the pattern '{original_column}.{Component}'
    (e.g., 'order_date.Year', 'order_date.Quarter'). The day of week is represented as an
    integer where 0 = Monday and 6 = Sunday, following pandas conventions.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing at least one column with date or datetime values.
        The DataFrame will be modified in place by adding new columns.
    date_column_name
        Name of the column containing date or datetime values from which to extract
        temporal components. The column must be datetime-compatible or convertible
        to datetime format.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with five additional columns appended:
          * {date_column_name}.Year: Four-digit year (e.g., 2023)
          * {date_column_name}.Quarter: Quarter number (1-4)
          * {date_column_name}.Month: Month number (1-12)
          * {date_column_name}.Day: Day of month (1-31)
          * {date_column_name}.DayOfWeek: Day of week (0=Monday, 6=Sunday)

    Examples
    --------
    # Add date components to a sales dataset
    import pandas as pd
    sales_df = pd.DataFrame({
        'order_date': pd.date_range('2023-01-01', periods=5, freq='D'),
        'amount': [100, 150, 200, 175, 225]
    })
    sales_df = AddDateNumberColumns(sales_df, 'order_date')
    # Result includes: order_date.Year, order_date.Quarter, order_date.Month, etc.

    # Useful for grouping by temporal periods
    transactions = pd.DataFrame({
        'transaction_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-01-22']),
        'revenue': [500, 750, 600]
    })
    transactions = AddDateNumberColumns(transactions, 'transaction_date')
    monthly_revenue = transactions.groupby('transaction_date.Month')['revenue'].sum()

    # Analyze weekly patterns in user activity
    user_activity = pd.DataFrame({
        'login_date': pd.date_range('2023-06-01', periods=30, freq='D'),
        'sessions': range(30)
    })
    user_activity = AddDateNumberColumns(user_activity, 'login_date')
    weekly_pattern = user_activity.groupby('login_date.DayOfWeek')['sessions'].mean()

    """
    
    # Extract year from date
    new_column_name = date_column_name + '.Year'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).year

    # Extract quarter from date
    new_column_name = date_column_name + '.Quarter'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).quarter

    # Extract month number from date
    new_column_name = date_column_name + '.Month'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).month

    # Extract day from date
    new_column_name = date_column_name + '.Day'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).day
    
    # Extract day of the week
    new_column_name = date_column_name + '.DayOfWeek'
    dataframe[new_column_name] = pd.DatetimeIndex(dataframe[date_column_name]).dayofweek
    print("Note: .DayOfWeek is 0-based starting on Monday (i.e. 0 = Monday, 6 = Sunday).")
    
    # Return dataframe
    return(dataframe)

