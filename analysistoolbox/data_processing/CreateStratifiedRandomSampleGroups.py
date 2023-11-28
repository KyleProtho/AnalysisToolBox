# Load packages
import numpy as np
import pandas as pd

# Declare function
def CreateStratifiedRandomSampleGroups(dataframe, 
                                       number_of_groups, 
                                       list_categorical_column_names, 
                                       random_seed=412):
    """
    Perform stratified random sampling on a DataFrame.

    Args:
        dataframe (pandas DataFrame): The input DataFrame.
        number_of_groups (int): The number of groups to create.
        list_categorical_column_names (list): A list of column names to use for stratification.
        random_seed (int, optional): The random seed for reproducibility (default is 412).

    Returns:
        pandas DataFrame: The DataFrame with an additional column 'group' indicating the group number.
    """
    # Ensure that the number of groups is a positive integer
    if number_of_groups <= 0:
        raise ValueError("Number of groups must be a positive integer (greater than 0)")
    
    # Ensure that the stratification variables are in the DataFrame
    if not all(col in dataframe.columns for col in list_categorical_column_names):
        raise ValueError("One or more stratification variables are not in the DataFrame")

    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Group by the stratification variables
    grouped = dataframe.groupby(list_categorical_column_names, group_keys=True)

    # For each group, assign records to n groups
    def assign_to_group(sub_df):
        sub_df['group'] = (sub_df.sample(frac=1, random_state=random_seed).reset_index(drop=True).index % number_of_groups) + 1
        return sub_df

    # Apply the function to each group and combine the results
    stratified_df = grouped.apply(assign_to_group).reset_index(drop=True)
    
    # Return the updated DataFrame
    return stratified_df

