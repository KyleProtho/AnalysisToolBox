# Load packages
import pandas as pd

# Declare function
def CleanTextColumns(dataframe):
    """
    Remove leading and trailing whitespace from all string columns in a DataFrame.

    This function automatically identifies and cleans all string-type (object dtype) columns
    in a DataFrame by applying the strip() operation to remove leading and trailing whitespace.
    This includes spaces, tabs, newlines, and other whitespace characters. The function
    modifies all string columns in place, making it a convenient one-step data cleaning operation.

    The function is particularly useful for:
      * Data import cleanup from CSV, Excel, or text files with formatting inconsistencies
      * Preprocessing data before joins or merges to avoid whitespace mismatch issues
      * Standardizing user-input data from web forms or surveys
      * Cleaning scraped web data with irregular spacing
      * Preparing data for string matching and deduplication
      * Database migration and ETL processes
      * General data quality improvement

    Only columns with object dtype (typically strings) are processed. Numeric, datetime,
    and other data types are left unchanged. This makes the function safe to apply to
    entire DataFrames without worrying about affecting non-text columns.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing one or more columns. String-type columns (object dtype)
        will be cleaned by removing leading and trailing whitespace. Other column types
        are not modified.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with all string columns cleaned. Leading and trailing whitespace
        is removed from all object dtype columns. The original DataFrame is modified in place
        and also returned.

    Examples
    --------
    # Clean messy imported data with extra spaces
    import pandas as pd
    messy_data = pd.DataFrame({
        'name': ['  Alice  ', 'Bob   ', '  Charlie'],
        'city': ['New York  ', '  Boston', '  Seattle  '],
        'age': [25, 30, 35]
    })
    clean_data = CleanTextColumns(messy_data)
    # name column becomes: ['Alice', 'Bob', 'Charlie']
    # city column becomes: ['New York', 'Boston', 'Seattle']
    # age column is unchanged (numeric type)

    # Prepare data for merging to avoid whitespace issues
    customers = pd.DataFrame({
        'customer_id': ['C001  ', '  C002', 'C003  '],
        'email': ['  alice@email.com', 'bob@email.com  ', '  charlie@email.com  ']
    })
    customers = CleanTextColumns(customers)
    # Ensures customer_id and email values match correctly during joins

    # Clean survey responses with inconsistent spacing
    survey = pd.DataFrame({
        'question_1': ['  Yes', 'No  ', '  Maybe  '],
        'question_2': ['Agree  ', '  Disagree', 'Neutral  '],
        'timestamp': pd.date_range('2023-01-01', periods=3)
    })
    survey = CleanTextColumns(survey)
    # Text columns cleaned, timestamp column unaffected

    # Process imported CSV with tab and newline characters
    raw_data = pd.DataFrame({
        'product': ['Widget\\t', '  Gadget\\n', '\\tDoohickey  '],
        'category': ['  Electronics\\n', 'Home\\t  ', '  Office  ']
    })
    clean_data = CleanTextColumns(raw_data)
    # Removes tabs, newlines, and spaces from all text entries

    """
    
    # Iterate over columns, and clean string-type columns
    for col_name in dataframe.columns:
        if dataframe.dtypes[col_name] == 'O':
            dataframe[col_name] = dataframe[col_name].str.strip()
            
    # Return dataframe
    return(dataframe)

