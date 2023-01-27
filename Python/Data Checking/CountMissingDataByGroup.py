# Load packages
import pandas as pd

def CountMissingDataByGroup(dataframe,
                            list_of_grouping_variables,
                            list_of_variables_to_count=None):
    """_summary_
    This function counts the number of records with missing data by group.

    Args:
        dataframe (_type_): Pandas dataframe
        list_of_grouping_variables (list): List of variables to group by
        list_of_variables_to_count (list, optional): List of variables to count missing data in. Defaults to None. If None, all variables are counted.
    """
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


# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# CountMissingDataByGroup(
#     dataframe=iris, 
#     list_of_grouping_variables=['species'],
# )
