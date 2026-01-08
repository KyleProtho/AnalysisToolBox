# Load packages
import pandas as pd
import os

# Declare function
def ImportDataFromFolder(folder_path,
                         force_column_names_to_match = True):
    """
    Batch import and combine all CSV and Excel files from a directory into single DataFrame.

    This function automates the process of importing multiple data files from a folder and
    combining them into a unified pandas DataFrame. It recursively processes all CSV (.csv)
    and Excel (.xlsx) files in the specified directory, reading each file and vertically
    concatenating (row-binding) them into a single dataset. The function handles column name
    mismatches either by forcing alignment to the first file's schema or raising an error,
    ensuring data consistency across the combined output.

    Batch data import and combination is essential for:
      * Consolidating monthly or periodic reports into annual datasets
      * Merging data exports from multiple sources or systems
      * Aggregating survey responses collected in separate files
      * Combining log files or transaction records across time periods
      * Creating master datasets from departmental or regional files
      * ETL (Extract, Transform, Load) workflows and data pipelines
      * Simplifying analysis of distributed data collections
      * Automating recurring data consolidation tasks

    The function uses pandas' read_csv() and read_excel() for file parsing, automatically
    detecting file types by extension. It validates column consistency across files, either
    enforcing uniform column names (using the first file as template) or raising an error
    if schemas don't match. The concatenation preserves all rows from all files, making it
    suitable for time-series data, append-only logs, and incremental datasets.

    Parameters
    ----------
    folder_path
        Absolute or relative path to the directory containing CSV and Excel files to import.
        All .csv and .xlsx files in this folder will be processed and combined.
    force_column_names_to_match
        Whether to automatically rename columns in subsequent files to match the first file's
        column names. If True, forces alignment by renaming columns (useful for files with
        minor naming variations). If False, raises ValueError when column names differ across
        files. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A single DataFrame containing all rows from all CSV and Excel files in the folder,
        vertically concatenated. Column names match the first file processed (if
        force_column_names_to_match is True) or are consistent across all files (if False).

    Raises
    ------
    ValueError
        If force_column_names_to_match is False and column names differ between files,
        indicating schema inconsistency.
    FileNotFoundError
        If the specified folder_path does not exist.
    PermissionError
        If the function lacks read permissions for the folder or any files within it.

    Notes
    -----
    * **File Types**: Only processes .csv and .xlsx files; ignores other formats
    * **Column Order**: Uses first file's column order as template when forcing matches
    * **Row Preservation**: Concatenates all rows; does not deduplicate or filter
    * **Index Reset**: May result in duplicate index values; consider resetting index
    * **Empty Folder**: Returns empty DataFrame if no CSV/Excel files found
    * **Encoding**: CSV files assumed to use default pandas encoding (UTF-8)

    Examples
    --------
    # Combine monthly sales reports
    annual_sales = ImportDataFromFolder('data/monthly_reports')
    # Imports all CSV and Excel files, forces column alignment

    # Strict schema validation
    survey_data = ImportDataFromFolder(
        folder_path='surveys/responses',
        force_column_names_to_match=False
    )
    # Raises error if any file has different columns

    # ETL workflow for transaction data
    import os
    
    # Combine transaction files from multiple sources
    transactions = ImportDataFromFolder('raw_data/transactions')
    
    # Reset index after concatenation
    transactions = transactions.reset_index(drop=True)
    
    # Add source tracking
    transactions['import_date'] = pd.Timestamp.now()
    
    # Save consolidated dataset
    transactions.to_csv('processed/all_transactions.csv', index=False)
    # Creates master transaction file from multiple sources

    # Quarterly report aggregation
    q1_data = ImportDataFromFolder('reports/Q1_2024')
    q2_data = ImportDataFromFolder('reports/Q2_2024')
    q3_data = ImportDataFromFolder('reports/Q3_2024')
    q4_data = ImportDataFromFolder('reports/Q4_2024')
    
    # Combine all quarters
    annual_data = pd.concat([q1_data, q2_data, q3_data, q4_data], ignore_index=True)
    # Creates annual dataset from quarterly folders

    # Regional data consolidation with validation
    try:
        regional_sales = ImportDataFromFolder(
            folder_path='data/regional_offices',
            force_column_names_to_match=False
        )
        print(f"Successfully imported {len(regional_sales)} records")
    except ValueError as e:
        print(f"Schema mismatch detected: {e}")
        # Handle inconsistent file structures
    # Validates schema consistency across regional files

    """
    
    # Create empty list to store dataframes
    df_list = []
    
    # Loop through all files in the folder
    for file in os.listdir(folder_path):
        # Check if file is a csv file
        if file.endswith(".csv"):
            # Import data from csv file
            df = pd.read_csv(folder_path + "/" + file)
            # Append dataframe to list
            df_list.append(df)
        # Check if file is an excel file
        elif file.endswith(".xlsx"):
            # Import data from excel file
            df = pd.read_excel(folder_path + "/" + file)
            # Append dataframe to list
            df_list.append(df)
    
    # Ensure that column names match across all dataframes
    for i in range(0, len(df_list)):
        # Check if column names match
        if list(df_list[i]) != list(df_list[0]):
            if force_column_names_to_match:
                # Rename columns
                df_list[i].columns = list(df_list[0])
            else:
                # Throw error
                raise ValueError("Column names do not match across all dataframes.")
    
    # Row bind all dataframes
    df_list = pd.concat(df_list, axis = 0)
    
    # Return list of dataframes
    return(df_list)

