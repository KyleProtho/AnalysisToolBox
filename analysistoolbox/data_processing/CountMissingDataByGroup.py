# Load packages
import pandas as pd

# Declare function
def CountMissingDataByGroup(dataframe,
                            list_of_grouping_columns,
                            list_of_columns_to_analyze=None):
    """
    This function counts the number of records with missing data by group.

    Args:
    dataframe (Pandas dataframe): Pandas dataframe
    list_of_grouping_columns (list): List of variables to group by
    list_of_columns_to_analyze (list, optional): List of variables to count missing data in. Defaults to None. If None, all variables are counted.

    Returns:
    Pandas dataframe: A Pandas dataframe with the number of records with missing data by group.
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

