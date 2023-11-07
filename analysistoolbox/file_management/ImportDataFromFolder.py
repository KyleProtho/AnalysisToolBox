# Load packages
import pandas as pd
import os

# Declare function
def ImportDataFromFolder(folder_path,
                         force_column_names_to_match = True):
    """
    Imports all CSV and Excel files from a folder and combines them into a single dataframe.
    Note: Each file should have the same column names.

    Args:
        folder_path (str): The path to the folder containing the csv and Excel files.
        force_column_names_to_match (bool, optional): Whether to force the column names to match across all files. Defaults to True.

    Returns:
        Pandas dataframe: A dataframe containing all the data from the csv and Excel files.
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

