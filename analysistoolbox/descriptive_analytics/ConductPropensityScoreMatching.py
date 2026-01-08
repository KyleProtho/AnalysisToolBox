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
    Perform propensity score matching to create balanced treatment and control groups.

    This function implements propensity score matching (PSM), a statistical technique used to
    estimate causal treatment effects in observational studies by reducing selection bias.
    PSM creates comparable groups by matching treated subjects with control subjects that have
    similar propensity scores—the probability of receiving treatment given observed covariates.
    The function uses the PsmPy package with logistic regression and k-nearest neighbors matching.

    Propensity score matching is essential for:
      * Estimating causal effects in observational studies without randomization
      * Reducing selection bias and confounding in treatment effect analysis
      * Creating balanced comparison groups for A/B testing post-hoc analysis
      * Medical research when randomized controlled trials are unethical or impractical
      * Policy evaluation and program impact assessment
      * Marketing campaign effectiveness measurement
      * Educational intervention analysis
      * Economic and social science research requiring causal inference

    The function automatically converts the grouping variable to binary format, calculates
    propensity scores using logistic regression, and performs k-nearest neighbors matching.
    It returns the original DataFrame enriched with propensity scores, logits, and matched
    subject IDs, enabling downstream causal effect estimation and balance diagnostics.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing subjects to match. Each row represents one subject
        with covariates, treatment assignment, and a unique identifier.
    subject_id_column_name
        Name of the column containing unique subject identifiers. Each value must be unique
        across all rows in the DataFrame.
    list_of_column_names_to_base_matching
        List of covariate column names to use for calculating propensity scores. These should
        be pre-treatment variables that influence both treatment assignment and outcomes.
    grouping_column_name
        Name of the column indicating treatment group membership. Must contain exactly two
        unique values (treatment and control).
    control_group_name
        The value in the grouping column that identifies the control group. All other values
        are treated as the treatment group.
    max_matches_per_subject
        Maximum number of control subjects to match with each treatment subject. Higher values
        increase statistical power but may reduce match quality. Defaults to 1.
    balance_groups
        Whether to balance the propensity score model by weighting observations. Set to False
        if matching fails with the default setting. Defaults to True.
    propensity_score_column_name
        Name for the new column containing propensity scores (probability of treatment).
        Defaults to 'Propensity Score'.
    propensity_logit_column_name
        Name for the new column containing propensity logits (log-odds of treatment).
        Defaults to 'Propensity Logit'.
    matched_id_column_name
        Name for the new column containing matched subject IDs. For subjects with multiple
        matches, multiple rows will be created. Defaults to 'Matched ID'.
    random_seed
        Random seed for reproducibility of the matching algorithm. Defaults to 412.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with three additional columns:
          * Propensity score column: Probability of treatment assignment (0-1)
          * Propensity logit column: Log-odds of treatment assignment
          * Matched ID column: Subject ID(s) of matched control/treatment subjects
        Subjects without matches will have NaN in the matched ID column. Subjects with
        multiple matches will appear in multiple rows.
    
    Examples
    --------
    # Medical trial: Match patients who received treatment with similar control patients
    import pandas as pd
    patient_df = pd.DataFrame({
        'patient_id': range(1, 101),
        'age': [25 + i for i in range(100)],
        'severity_score': [50 + i % 30 for i in range(100)],
        'comorbidities': [i % 3 for i in range(100)],
        'received_treatment': ['Treatment' if i % 3 == 0 else 'Control' for i in range(100)]
    })
    matched_df = ConductPropensityScoreMatching(
        patient_df,
        subject_id_column_name='patient_id',
        list_of_column_names_to_base_matching=['age', 'severity_score', 'comorbidities'],
        grouping_column_name='received_treatment',
        control_group_name='Control',
        max_matches_per_subject=1
    )
    # Analyze treatment effect using matched pairs

    # Marketing campaign: Match customers who saw ad with similar non-exposed customers
    customer_df = pd.DataFrame({
        'customer_id': [f'C{i:04d}' for i in range(200)],
        'purchase_history': [i % 10 for i in range(200)],
        'engagement_score': [30 + i % 50 for i in range(200)],
        'demographic_segment': [i % 5 for i in range(200)],
        'saw_campaign': [1 if i % 4 == 0 else 0 for i in range(200)]
    })
    campaign_matched = ConductPropensityScoreMatching(
        customer_df,
        subject_id_column_name='customer_id',
        list_of_column_names_to_base_matching=['purchase_history', 'engagement_score', 'demographic_segment'],
        grouping_column_name='saw_campaign',
        control_group_name=0,
        max_matches_per_subject=2,
        balance_groups=True,
        random_seed=42
    )
    # Estimate campaign lift using matched controls

    # Educational intervention: Match students with multiple controls
    student_df = pd.DataFrame({
        'student_id': range(1, 151),
        'prior_gpa': [2.0 + (i % 20) / 10 for i in range(150)],
        'attendance_rate': [70 + i % 30 for i in range(150)],
        'socioeconomic_status': [i % 4 for i in range(150)],
        'program_participant': ['Yes' if i % 5 == 0 else 'No' for i in range(150)]
    })
    education_matched = ConductPropensityScoreMatching(
        student_df,
        subject_id_column_name='student_id',
        list_of_column_names_to_base_matching=['prior_gpa', 'attendance_rate', 'socioeconomic_status'],
        grouping_column_name='program_participant',
        control_group_name='No',
        max_matches_per_subject=3,
        propensity_score_column_name='PS',
        matched_id_column_name='Matched_Student'
    )
    # Evaluate program impact with 1:3 matching ratio

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
