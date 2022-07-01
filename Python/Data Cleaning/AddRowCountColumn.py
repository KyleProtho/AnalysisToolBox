# Load packages
import pandas as pd

# Declare function
def AddRowCountColumn(dataframe,
                      list_of_order_columns,
                      list_of_grouping_variables,
                      list_of_ascending_order_args=None,
                      row_count_column_name='Row Count'):
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
