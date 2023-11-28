# Load packages
import pandas as pd

# Declare function
def CreateRandomSampleGroups(dataframe,
                             number_of_groups=2, 
                             random_seed=412):
    """
    Shuffle the DataFrame rows, assign each record to one of n groups, then restore the original order and return the updated DataFrame.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        number_of_groups (int, optional): The number of groups to create. Defaults to 2.
        random_seed (int, optional): The random seed for reproducibility. Defaults to None.

    Returns:
        pandas.DataFrame: The updated DataFrame with an additional column 'group' indicating the group number.
    """
    # Ensure that the number of groups is a positive integer
    if number_of_groups <= 0:
        raise ValueError("Number of groups must be a positive integer (greater than 0)")

    # Shuffle the DataFrame
    shuffled_df = dataframe.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Assign groups to the shuffled DataFrame
    group_numbers = [i % number_of_groups + 1 for i in range(len(shuffled_df))]
    shuffled_df['group'] = group_numbers
    
    # Return the updated DataFrame
    return shuffled_df

