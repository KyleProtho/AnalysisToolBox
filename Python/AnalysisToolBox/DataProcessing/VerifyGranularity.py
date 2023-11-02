# Load packages
import pandas as pd
from IPython.display import display, Markdown

# Declare function
def VerifyGranularity(dataframe,
                      list_of_key_columns,
                      set_key_as_index=True,
                      print_as_markdown=True):
    """
    Verifies the granularity of a given dataframe based on a list of key columns.
    This function creates a key column from the provided list of key columns and checks if the number of rows in the dataframe equals the number of distinct values in the key column. If the counts do not match, a warning message is printed. 
    The function can optionally set the key column as the dataframe's index and print the results as markdown.

    Args:
        dataframe (pd.DataFrame): The dataframe to verify.
        list_of_key_columns (list): The list of columns to use for creating the key column.
        set_key_as_index (bool, optional): If True, sets the key column as the dataframe's index. Defaults to True.
        print_as_markdown (bool, optional): If True, prints the results as markdown. Defaults to True.

    Returns:
        pd.DataFrame: The updated dataframe with the key column set as the index, if requested.
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


# # Test function
# import numpy as np
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# # Create id column based on row number
# iris['id'] = np.arange(len(iris))
# # Test VerifyGranularity function
# VerifyGranularity(iris, ['id'])
