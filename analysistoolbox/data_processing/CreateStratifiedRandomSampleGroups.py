# Load packages
import numpy as np
import pandas as pd

# Declare function
def CreateStratifiedRandomSampleGroups(dataframe, 
                                       number_of_groups, 
                                       list_categorical_column_names, 
                                       random_seed=412):
    """
    Partition a DataFrame into balanced groups using stratified random assignment.

    This function assigns each record in a DataFrame to one of $n$ groups while ensuring 
    that the distribution of specific categorical variables remains consistent across 
    all groups. By stratifying on key features, the function prevents accidental 
    biases that can occur with simple random assignment, ensuring that each group is 
    statistically representative of the original population.

    The function is particularly useful for:
      * Designing A/B tests where groups must be balanced by demographics (e.g., age or region)
      * Creating cross-validation folds that preserve class balance for machine learning
      * Designing clinical or social research trials where treatment groups must be comparable
      * Splitting marketing segments to ensure representative testing across customer tiers
      * Educational research where student groups must be balanced by prior performance
      * Validating model performance across different sub-populations in a balanced way

    The assignment process is reproducible when a `random_seed` is provided. The 
    resulting 'group' column contains integers ranging from 1 to `number_of_groups`.

    Parameters
    ----------
    dataframe
        The input pandas DataFrame containing the records to be partitioned.
    number_of_groups
        The number of discrete groups to create. Records within each stratum are 
        distributed as evenly as possible across these groups.
    list_categorical_column_names
        A list of column names in the DataFrame to use for stratification. The 
        function ensures that the proportions of these categories are roughly 
        equal in every resulting group.
    random_seed
        The seed value for the random number generator to ensure reproducibility.
        Defaults to 412.

    Returns
    -------
    pd.DataFrame
        The stratified DataFrame with an additional 'group' column (integer values) 
        indicating the assigned group number. Note that the DataFrame may be 
        resorted by the stratification variables.

    Examples
    --------
    # Create balanced A/B test groups based on gender and customer tier
    import pandas as pd
    users = pd.DataFrame({
        'user_id': range(1, 13),
        'tier': ['Gold', 'Gold', 'Silver', 'Silver', 'Bronze', 'Bronze'] * 2,
        'gender': ['M', 'F'] * 6
    })
    stratified_groups = CreateStratifiedRandomSampleGroups(
        users, 
        number_of_groups=2, 
        list_categorical_column_names=['tier', 'gender']
    )
    # Each of the 2 groups will have a representative mix of Tier and Gender.

    # Partition data into 3 folds while preserving class balance
    data = pd.DataFrame({
        'feature': range(90),
        'label': ['Target'] * 30 + ['Control'] * 60
    })
    folds = CreateStratifiedRandomSampleGroups(
        data, 
        number_of_groups=3, 
        list_categorical_column_names=['label']
    )
    # Each fold will contain exactly 10 'Target' and 20 'Control' records.

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

