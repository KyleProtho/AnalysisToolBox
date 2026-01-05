# Load packages
import pandas as pd
from IPython.display import display, Markdown

# Declare function
def VerifyGranularity(dataframe,
                      list_of_key_columns,
                      set_key_as_index=True,
                      print_as_markdown=True):
    """
    Verify and enforce a unique level of granularity for a DataFrame.

    This function validates whether a specified combination of columns uniquely
    identifies each row in the DataFrame. It creates a composite 'Dataset Key' by
    concatenating the specified columns and compares the total row count against
    the number of unique keys. This is a critical step in data validation to ensure
    integrity before joining datasets or performing aggregations.

    The function is particularly useful for:
      * Validating primary key assumptions in new datasets
      * Ensuring data integrity before performing table joins
      * Identifying unexpected duplicates in transactional logs
      * Documenting the grain/level of analysis for a dataset
      * Preparing DataFrames for indexing in specialized time-series or panel data
      * Debugging ETL pipelines where row duplication may have occurred

    The function provides visual feedback (check marks or warnings) and can
    optionally transform the composite key into the DataFrame's primary index.

    Parameters
    ----------
    dataframe
        The pandas DataFrame to investigate.
    list_of_key_columns
        A list of column names that are expected to form a unique identifier for
        each row. Columns will be concatenated with a ' -- ' separator.
    set_key_as_index
        If True, the calculated composite key will be set as the DataFrame's
        index, and the temporary 'Dataset Key' column will be removed.
        Defaults to True.
    print_as_markdown
        If True, results and warnings are formatted using IPython Markdown for
        rich display in Jupyter Notebooks. If False, standard print statements
        are used. Defaults to True.

    Returns
    -------
    pd.DataFrame
        If `set_key_as_index` is True, returns the modified DataFrame with the
        composite key as its index. Otherwise, returns the updated DataFrame
        (though the original may be modified in-place).

    Examples
    --------
    # Verify granularity of sales data by Date and StoreID
    import pandas as pd
    sales = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-01', '2023-01-02'],
        'StoreID': [101, 102, 101],
        'Revenue': [5000, 6200, 4800]
    })
    sales_indexed = VerifyGranularity(
        sales, 
        ['Date', 'StoreID'], 
        set_key_as_index=True
    )
    # Prints success message and sets index to '2023-01-01 -- 101', etc.

    # Detect duplicates in customer records
    customers = pd.DataFrame({
        'Email': ['test@abc.com', 'test@abc.com', 'user2@abc.com'],
        'Name': ['John', 'John Doe', 'Jane']
    })
    VerifyGranularity(customers, ['Email'], print_as_markdown=False)
    # Prints a warning: counts will not match due to the duplicate Email.

    """
    
    # Create key column from list of key columns
    first_col = list_of_key_columns[0]
    dataframe['Dataset Key'] = dataframe[first_col].astype(str)
    
    # If more than 1 column listed, concatenate columns together
    if len(list_of_key_columns) > 1:
        for key_col in list_of_key_columns[1:]:
            dataframe['Dataset Key'] = dataframe['Dataset Key'].astype(str) + " -- " + dataframe[key_col].astype(str)
    
    # Get row count and distinct key count
    row_count = len(dataframe.index)
    distinct_key_count = len(pd.unique(dataframe['Dataset Key']))
    
    # Create row count string, format the count with thousands separator, and print it
    row_count_string = "Number of rows in dataframe: " + "{:,}".format(row_count)
    # Create distinct key count string, format the count with thousands separator, and print it
    distinct_key_count_string = "Distinct count of Dataset Keys: " + "{:,}".format(distinct_key_count)
    
    # Show granularity results as markdown if requested
    if print_as_markdown:
        # Show the results using check mark emoji if row count equals distinct key count
        display(Markdown(row_count_string))
        display(Markdown(distinct_key_count_string))
        if row_count != distinct_key_count:
            display(Markdown("⚠️ WARNING: Unique combination of columns you listed does not equal number of rows in dataset. Try a new combination of columns or see if dataset has duplicates."))
        else:
            display(Markdown("✅ SUCCESS: Dataset granularity is verified using the columns you listed."))
    else:
        print(row_count_string)
        print(distinct_key_count_string)
        if row_count != distinct_key_count:
            print("WARNING: Unique combination of columns you listed does not equal number of rows in dataset. Try a new combination of columns or see if dataset has duplicates.")
        else:
            print("SUCCESS: Dataset granularity is verified using the columns you listed.")
        
    # If Key column not being kept, drop it. Otherwise, move Key column to front of DataFrame
    if set_key_as_index != True:
        dataframe = dataframe.drop(columns=['Dataset Key'])
    else:
        dataframe.index = dataframe['Dataset Key']
        dataframe = dataframe.drop(columns=['Dataset Key'])
        return(dataframe)

