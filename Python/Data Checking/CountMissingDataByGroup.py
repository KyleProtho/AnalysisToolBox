# Load packages
import pandas as pd

def CountMissingDataByGroup(dataframe,
                            list_of_grouping_variables,
                            list_of_variables_to_count=None):
    # Select variables to count missing data for
    if list_of_variables_to_count is not None:
        dataframe = dataframe[list_of_grouping_variables + list_of_variables_to_count]
    
    # Count missing data by group
    df_missing_by_group = dataframe.groupby(
        list_of_grouping_variables,
        dropna=False).apply(lambda x: x.isnull().sum()
    )
        
    # Get row count by group
    df_row_count_by_group = dataframe.groupby(
        list_of_grouping_variables,
        dropna=False
    ).size()
    df_missing_by_group['Row count'] = df_row_count_by_group
    del(df_row_count_by_group)
    
    # Set row count as first column
    first_column = df_missing_by_group.pop('Row count')
    df_missing_by_group.insert(0, 'Row count', first_column)
    
    # Return missing data by group summary
    return(df_missing_by_group)
    