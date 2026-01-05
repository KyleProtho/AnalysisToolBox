# Load packages
import pandas as pd
import numpy as np

# Declare function
def AddLeadingZeros(dataframe,
                    column_name,
                    fixed_length=None,
                    add_as_new_column=False):
    """
    Pad numeric or string values with leading zeros to achieve a fixed string length.

    This function standardizes the format of values in a DataFrame column by adding leading
    zeros to ensure all values reach a specified length. This is commonly needed for codes,
    identifiers, ZIP codes, account numbers, and other fields that require fixed-width
    formatting for consistency, sorting, or integration with external systems.

    The function is particularly useful for:
      * Formatting ZIP codes (e.g., '02134' instead of '2134')
      * Standardizing ID numbers and account codes
      * Preparing data for systems that require fixed-width text files
      * Ensuring proper alphanumeric sorting of numeric strings
      * Creating consistently formatted reports and exports
      * Data cleaning and normalization tasks

    When no fixed length is specified, the function automatically determines the appropriate
    length by finding the longest value in the column. The function preserves NaN values
    and removes trailing '.0' artifacts that may appear when converting numeric types.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the column to be formatted with leading zeros.
        The DataFrame will be modified based on the add_as_new_column parameter.
    column_name
        Name of the column to pad with leading zeros. The column can contain numeric
        or string values. Values will be converted to strings before padding.
    fixed_length
        Target length for all values after padding with leading zeros. If None, the
        function automatically uses the length of the longest value in the column.
        Defaults to None.
    add_as_new_column
        If True, creates a new column named '{column_name} - with leading 0s' containing
        the padded values, leaving the original column unchanged. If False, updates the
        original column in place. Defaults to False.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with either the original column updated (if add_as_new_column=False)
        or a new column added (if add_as_new_column=True). All values are padded to the
        specified or automatically determined length. NaN values are preserved.

    Examples
    --------
    # Format ZIP codes with leading zeros
    import pandas as pd
    addresses = pd.DataFrame({
        'zip_code': [2134, 90210, 10001, 501]
    })
    addresses = AddLeadingZeros(addresses, 'zip_code', fixed_length=5)
    # zip_code column becomes: ['02134', '90210', '10001', '00501']

    # Standardize employee IDs without specifying length (auto-detects max length)
    employees = pd.DataFrame({
        'emp_id': [1, 42, 999, 1234]
    })
    employees = AddLeadingZeros(employees, 'emp_id')
    # emp_id column becomes: ['0001', '0042', '0999', '1234']

    # Add leading zeros as a new column, preserving original values
    products = pd.DataFrame({
        'product_code': ['A1', 'A22', 'A333']
    })
    products = AddLeadingZeros(products, 'product_code', fixed_length=6, add_as_new_column=True)
    # Creates new column 'product_code - with leading 0s': ['0000A1', '000A22', '00A333']

    # Handle missing values gracefully
    accounts = pd.DataFrame({
        'account_number': [123, None, 456, 78]
    })
    accounts = AddLeadingZeros(accounts, 'account_number', fixed_length=4)
    # account_number column: ['0123', NaN, '0456', '0078']

    """
    
    # If fixed length not specified, set the longest string as the fixed length
    if fixed_length == None:
        fixed_length = max(dataframe[column_name].astype(str).str.len())
    
    # If adding as new column, change the column name
    if add_as_new_column:
        new_column_name = column_name + ' - with leading 0s'
        dataframe[new_column_name] = np.where(
            dataframe[column_name].isna(),
            np.nan,
            dataframe[column_name].astype(str).str.zfill(fixed_length)
        )
        dataframe[new_column_name] = dataframe[new_column_name].str.replace(".0", "",
                                                                            regex=False)
    else:
        dataframe[column_name] = np.where(
            dataframe[column_name].isna(),
            np.nan,
            dataframe[column_name].astype(str).str.zfill(fixed_length)
        )
        dataframe[column_name] = dataframe[column_name].str.replace(".0", "",
                                                                    regex=False)
    
    # Return updated dataframe
    return(dataframe)

