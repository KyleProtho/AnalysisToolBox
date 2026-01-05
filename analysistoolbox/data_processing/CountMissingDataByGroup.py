# Load packages
import pandas as pd

# Declare function
def CountMissingDataByGroup(dataframe,
                            list_of_grouping_columns,
                            list_of_columns_to_analyze=None):
    """
    Calculate the count of missing values within specified groups.

    This function aggregates a DataFrame by one or more grouping columns and calculates
    the number of missing (NaN/null) values for either all columns or a specific subset
    of columns. It also provides a total row count for each group, allowing for easy
    calculation of missingness percentages.

    The function is particularly useful for:
      * Identifying data quality issues across different segments or categories
      * Analyzing survey data to find patterns in non-response
      * Pre-processing data to determine if certain groups should be excluded due to sparse information
      * Validating data integrity after merging datasets from multiple sources
      * Monitoring data pipelines for unexpected drops in data completeness
      * Reporting on audit and compliance metrics for required fields

    The function preserves information about groups that may contain missing values themselves
    by setting `dropna=False` during the grouping operation.

    Parameters
    ----------
    dataframe
        The pandas DataFrame to analyze.
    list_of_grouping_columns
        A list of column names to group the data by. These variables define the
        segments for which missing data will be counted.
    list_of_columns_to_analyze
        A list of column names in which to count missing values. If None, the function
        counts missing values for all columns in the DataFrame. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame indexed by the grouping columns. It contains a 'Row count'
        column representing the total number of records per group, followed by columns
        showing the count of missing values for each analyzed variable.

    Examples
    --------
    # Analyze missing demographic data by region
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        'Region': ['East', 'East', 'West', 'West', 'North', 'East'],
        'Age': [25, np.nan, 30, np.nan, 40, 22],
        'Income': [50000, 60000, np.nan, np.nan, 75000, np.nan]
    })
    missing_summary = CountMissingDataByGroup(df, ['Region'])
    # Results will show Row count, Age missingness, and Income missingness per Region

    # Focus on specific columns for a multi-level group
    survey_data = pd.DataFrame({
        'Year': [2021, 2021, 2022, 2022],
        'Branch': ['A', 'A', 'B', 'A'],
        'Revenue': [100, np.nan, 200, 150],
        'Employees': [5, 10, np.nan, 8]
    })
    branch_completeness = CountMissingDataByGroup(
        survey_data, 
        ['Year', 'Branch'], 
        ['Revenue']
    )
    # Shows Revenue completeness indexed by Year and Branch

    """
    
    # Select variables to count missing data for
    if list_of_columns_to_analyze is not None:
        dataframe = dataframe[list_of_grouping_columns + list_of_columns_to_analyze]
    
    # Count missing data by group
    df_missing_by_group = dataframe.groupby(
        list_of_grouping_columns,
        dropna=False).apply(lambda x: x.isnull().sum()
    )
        
    # Get row count by group
    df_row_count_by_group = dataframe.groupby(
        list_of_grouping_columns,
        dropna=False
    ).size()
    df_missing_by_group['Row count'] = df_row_count_by_group
    del(df_row_count_by_group)
    
    # Set row count as first column
    first_column = df_missing_by_group.pop('Row count')
    df_missing_by_group.insert(0, 'Row count', first_column)
    
    # Return missing data by group summary
    return(df_missing_by_group)

