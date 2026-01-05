# Load packages
import pandas as pd

# Declare function
def AddRowCountColumn(dataframe,
                      list_of_grouping_variables,
                      list_of_order_columns,
                      list_of_ascending_order_args=None,
                      row_count_column_name='Row Count'):
    """
    Add a sequential row count column within groups based on specified sorting criteria.

    This function enriches a DataFrame by adding a numbered sequence column that counts rows
    within each group. The groups are defined by grouping variables, and the numbering follows
    a custom sort order specified by order columns. This creates a ranked or indexed sequence
    within each group, similar to SQL window functions like ROW_NUMBER().

    The function is particularly useful for:
      * Ranking records within groups (e.g., top 3 products per category)
      * Creating sequential identifiers within partitions
      * Implementing pagination or batch processing logic
      * Identifying first/last occurrences within groups
      * Generating ordered indices for time series analysis within entities
      * Implementing custom sorting and ranking logic

    The row count starts at 0 for the first row in each group (following pandas convention)
    and increments by 1 for each subsequent row. The sorting order can be customized for
    each column independently using ascending/descending flags.

    Parameters
    ----------
    dataframe
        A pandas DataFrame to which the row count column will be added. The DataFrame
        will be sorted according to the specified order columns and groups.
    list_of_grouping_variables
        List of column names that define the groups. Rows with the same values across
        these columns will be grouped together, and row counting will restart at 0 for
        each unique group combination.
    list_of_order_columns
        List of column names that determine the sorting order within each group. The
        row count will follow this sort order. Multiple columns create a hierarchical
        sort (first column has priority, then second, etc.).
    list_of_ascending_order_args
        List of boolean values corresponding to each column in list_of_order_columns,
        specifying whether to sort in ascending (True) or descending (False) order.
        If None, all columns are sorted in ascending order. Defaults to None.
    row_count_column_name
        Name for the new row count column that will be added to the DataFrame.
        Defaults to 'Row Count'.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column containing the row count within
        each group. The DataFrame is sorted by the grouping variables and the row count
        column. Row counts start at 0 for the first row in each group.

    Examples
    --------
    # Rank sales transactions by amount within each store
    import pandas as pd
    sales = pd.DataFrame({
        'store': ['A', 'A', 'A', 'B', 'B', 'B'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'amount': [100, 250, 150, 300, 200, 180]
    })
    sales = AddRowCountColumn(
        sales,
        list_of_grouping_variables=['store'],
        list_of_order_columns=['amount'],
        list_of_ascending_order_args=[False],  # Highest amounts first
        row_count_column_name='Sales Rank'
    )
    # Each store gets rows ranked by amount (0 = highest, 1 = second, etc.)

    # Number customer orders chronologically within each customer
    orders = pd.DataFrame({
        'customer_id': [101, 101, 101, 102, 102],
        'order_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-01-05', '2023-02-12'],
        'order_total': [50, 75, 60, 120, 90]
    })
    orders = AddRowCountColumn(
        orders,
        list_of_grouping_variables=['customer_id'],
        list_of_order_columns=['order_date'],
        row_count_column_name='Order Number'
    )
    # Creates Order Number: 0, 1, 2 for customer 101 and 0, 1 for customer 102

    # Multi-level sorting: rank employees by department and hire date
    employees = pd.DataFrame({
        'department': ['Sales', 'Sales', 'IT', 'IT', 'Sales'],
        'hire_date': ['2020-01-01', '2019-05-15', '2021-03-10', '2018-07-20', '2022-02-01'],
        'salary': [60000, 55000, 70000, 80000, 58000]
    })
    employees = AddRowCountColumn(
        employees,
        list_of_grouping_variables=['department'],
        list_of_order_columns=['hire_date'],
        row_count_column_name='Seniority Rank'
    )
    # Ranks employees within each department by hire date (earliest = 0)

    # Default ascending order when list_of_ascending_order_args is None
    products = pd.DataFrame({
        'category': ['Electronics', 'Electronics', 'Clothing', 'Clothing'],
        'price': [999, 299, 49, 79]
    })
    products = AddRowCountColumn(
        products,
        list_of_grouping_variables=['category'],
        list_of_order_columns=['price']
    )
    # Automatically uses ascending order for price (lowest = 0)

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

