# Load packages
import pandas as pd

# Declare function
def CreateRandomSampleGroups(dataframe,
                             number_of_groups=2, 
                             random_seed=412):
    """
    Randomly assign DataFrame records to a specified number of groups.

    This function shuffles the rows of a DataFrame and assigns each record to one of $n$
    groups in a balanced manner. This is a fundamental technique for creating
    unbiased subsets of data, ensuring that each group has a roughly equal number
    of observations and that the assignment is independent of the original data order.

    The function is particularly useful for:
      * A/B testing and randomized controlled trials (assigning users to treatment/control)
      * K-fold cross-validation (manually partitioning data into folds)
      * Pilot studies where a representative sample is needed from a larger population
      * Stress testing systems by splitting load into different processing buckets
      * Marketing campaigns where different offers are sent to random customer segments
      * Creating development, validation, and test sets with custom ratios

    The assignment process is reproducible when a `random_seed` is provided. The 
    resulting 'group' column contains integers ranging from 1 to `number_of_groups`.

    Parameters
    ----------
    dataframe
        The input pandas DataFrame containing the records to be partitioned.
    number_of_groups
        The number of discrete groups to create. Records are distributed as evenly
        as possible across these groups. Defaults to 2.
    random_seed
        The seed value for the random number generator to ensure reproducibility.
        Defaults to 412.

    Returns
    -------
    pd.DataFrame
        The shuffled DataFrame with an additional 'group' column (integer values) 
        indicating the assigned group number.

    Examples
    --------
    # Assign customers to Treatment (1) and Control (2) groups
    import pandas as pd
    customers = pd.DataFrame({
        'customer_id': range(1001, 1007),
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank']
    })
    experimental_groups = CreateRandomSampleGroups(
        customers, 
        number_of_groups=2, 
        random_seed=42
    )
    # Returns shuffled DataFrame with 'group' column containing 1s and 2s

    # Split a dataset into 5 groups for cross-validation
    data = pd.DataFrame({'feature': range(100)})
    folds = CreateRandomSampleGroups(data, number_of_groups=5)
    group_counts = folds['group'].value_counts()
    # Each group will have exactly 20 records

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

