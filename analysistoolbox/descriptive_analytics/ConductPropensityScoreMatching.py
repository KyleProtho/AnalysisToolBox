# Load packages
import pandas as pd

# Delcare function
def ConductPropensityScoreMatching(dataframe,
                                   subject_id_column_name,
                                   list_of_column_names_to_base_matching,
                                   grouping_column_name,
                                   control_group_name,
                                   max_matches_per_subject=1,
                                   balance_groups=True,
                                   propensity_score_column_name="Propensity Score",
                                   propensity_logit_column_name="Propensity Logit",
                                   matched_id_column_name="Matched ID",
                                   random_seed=412):
    """
    This function conducts propensity score matching on a dataframe. The function uses the PsmPy package to perform the matching. 
    The function returns the original dataframe with the propensity scores, propensity logits, and matched IDs appended to the dataframe. 
    The function also returns the matched dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe to conduct the matching on.
        subject_id_column_name (str): The name of the column that contains the subject IDs.
        list_of_column_names_to_base_matching (list): A list of column names to base the matching on.
        grouping_column_name (str): The name of the column that contains the grouping values.
        control_group_name (str): The name of the control group.
        max_matches_per_subject (int, optional): The maximum number of matches per subject. Defaults to 1.
        balance_groups (bool, optional): Whether to balance the groups. Defaults to True.
        propensity_score_column_name (str, optional): The name of the column that contains the propensity scores. Defaults to "Propensity Score".
        propensity_logit_column_name (str, optional): The name of the column that contains the propensity logits. Defaults to "Propensity Logit".
        matched_id_column_name (str, optional): The name of the column that contains the matched IDs. Defaults to "Matched ID".
        random_seed (int, optional): The random seed to use for the matching. Defaults to 412.

    Raises:
        ValueError: The grouping column must have only two unique values.
        ValueError: The control group name must be one of the unique values in the grouping column.
        ValueError: The subject ID column must have unique values.
        ValueError: The treatment group must have at least one row.
        ValueError: The control group must have at least one row.

    Returns:
        pd.DataFrame: The matched dataframe.
    """
    
    # Lazy load uncommon packages
    from psmpy import PsmPy
    
    # Ensure that the grouping column has only two unique values
    if len(dataframe[grouping_column_name].unique()) != 2:
        raise ValueError("The grouping column must have only two unique values.")
    
    # Ensure that the control group name is one of the unique values in the grouping column
    if control_group_name not in dataframe[grouping_column_name].unique():
        raise ValueError("The control group name must be one of the unique values in the grouping column.")
    
    # Ensure that the subject ID column has unique values
    if len(dataframe[subject_id_column_name].unique()) != len(dataframe):
        raise ValueError("The subject ID column must have unique values.")
    
    # Split the dataframe into the treatment and control groups
    treatment_group = dataframe[dataframe[grouping_column_name] != control_group_name]
    control_group = dataframe[dataframe[grouping_column_name] == control_group_name]
    
    # Ensure that the treatment and control groups have at least one row
    if len(treatment_group) == 0:
        raise ValueError("The treatment group must have at least one row.")
    if len(control_group) == 0:
        raise ValueError("The control group must have at least one row.")
    
    # Select the necessary columns for the matching
    dataframe_psm = dataframe[[subject_id_column_name] + list_of_column_names_to_base_matching + [grouping_column_name]]
    
    # Drop any rows with missing values in grouping column or subject ID column
    dataframe_psm = dataframe_psm.dropna(subset=[subject_id_column_name, grouping_column_name])
    
    # If grouping column is not binary, then convert it to binary
    if dataframe_psm[grouping_column_name].unique().tolist() != [1, 0] or dataframe_psm[grouping_column_name].unique().tolist() != [0, 1]:
        dataframe_psm[grouping_column_name] = dataframe_psm[grouping_column_name].apply(lambda x: 0 if x == control_group_name else 1)
    
    # Initalize a PSMatcher object
    psm = PsmPy(
        dataframe_psm, 
        treatment=grouping_column_name, 
        indx=subject_id_column_name,
        exclude=[],
        seed=random_seed,
    )
    
    # Calculate logistic propensity scores/logits
    psm.logistic_ps(
        balance=balance_groups
    )
    
    # Perform KNN matching 1:max_matches_per_subject
    try:
        psm.knn_matched_12n(
            matcher='propensity_logit', 
            how_many=max_matches_per_subject
        )
    except Exception as e:
        raise ValueError("The propensity score matching failed. \n" + e + "\n\nTry setting balance_groups to False.")
    
    # Create a dataframe of the matched subjects
    dataframe_matched = psm.df_matched
    
    # Merge the matched subjects with the original dataframe
    dataframe_matched = dataframe_matched[[subject_id_column_name, "propensity_score", "propensity_logit"]]
    dataframe_matched.columns = [subject_id_column_name, propensity_score_column_name, propensity_logit_column_name]
    dataframe = pd.merge(
        dataframe,
        dataframe_matched,
        on=subject_id_column_name,
        how="left"
    )
    
    # Get the matched subject IDs
    dataframe_matched = psm.matched_ids
    
    # For multiple matches, the matched_ids dataframe will have multiple columns
    # Rename columns to be consistent
    new_cols = [subject_id_column_name] + [f"{matched_id_column_name}_{i+1}" for i in range(max_matches_per_subject)]
    dataframe_matched.columns = new_cols
    
    # Melt the matched ID columns into a single column
    id_vars = [subject_id_column_name]
    value_vars = [col for col in dataframe_matched.columns if col.startswith(matched_id_column_name)]
    dataframe_matched = dataframe_matched.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='match_number',
        value_name=matched_id_column_name
    )
    
    # Drop the match_number column and remove any rows where matched_id is None
    dataframe_matched = dataframe_matched.drop('match_number', axis=1)
    dataframe_matched = dataframe_matched.dropna(subset=[matched_id_column_name])
    
    # Create reciprocal matches (if A is matched to B, B should be matched to A)
    data_matched_2 = dataframe_matched[[matched_id_column_name, subject_id_column_name]]
    data_matched_2.columns = [subject_id_column_name, matched_id_column_name]
    
    # Row bind the flipped matched subject IDs with the original matched subject IDs
    data_matched = pd.concat([dataframe_matched, data_matched_2], axis=0)
    del(data_matched_2)
    
    # Remove any duplicate matches
    data_matched = data_matched.drop_duplicates()
    
    # Merge the matched subject IDs with the original dataframe
    dataframe = dataframe.merge(
        data_matched,
        on=subject_id_column_name,
        how="left"
    )
    del(data_matched)
    
    # Return the matched dataframe
    return dataframe
