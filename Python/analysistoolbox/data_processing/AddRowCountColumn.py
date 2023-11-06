# Load packages
import pandas as pd

# Declare function
def AddRowCountColumn(dataframe,
                      list_of_grouping_variables,
                      list_of_order_columns,
                      list_of_ascending_order_args=None,
                      row_count_column_name='Row Count'):
    """
    This function adds a row count column to a dataset based on a list of order columns, a list of grouping variables, and a list of ascending order arguments.
    
    Args:
    dataframe (Pandas dataframe): Pandas dataframe
    list_of_grouping_variables (list): The list of columns to group by.
    list_of_order_columns (list): The list of columns to order by.
    list_of_ascending_order_args (list, optional): The list of ascending order arguments. Defaults to None. If None, all columns are ordered in ascending order.
    row_count_column_name (str, optional): The name of the now row count column. Defaults to 'Row Count'.

    Returns:
    Pandas dataframe: An updated Pandas dataframe with a row count column.
    """
    
    # Order dataframe by order columns
    if (list_of_ascending_order_args == None):
        dataframe = dataframe.sort_values(
            list_of_order_columns
        )
    else:
        dataframe = dataframe.sort_values(
            list_of_order_columns,
            ascending=list_of_ascending_order_args
        )
        
    # Add row count
    dataframe = dataframe.groupby(
        list_of_grouping_variables
    ).cumcount()
    
    # Resort by grouping columns and row order column
    list_of_order_columns.append(row_count_column_name)
    dataframe = dataframe.sort_values(
        list_of_order_columns
    )
    
    # Return dataframe
    return(dataframe)

