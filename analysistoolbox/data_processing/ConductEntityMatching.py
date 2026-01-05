# Load packages
import numpy as np
import pandas as pd

# Declare function
def ConductEntityMatching(dataframe_1, 
                          dataframe_1_primary_key, 
                          dataframe_2, 
                          dataframe_2_primary_key,
                          levenshtein_distance_filter=None,
                          match_score_threshold=95,
                          columns_to_compare=None,
                          match_methods=['Partial Token Set Ratio', 'Weighted Ratio']):
    """
    Match and link records across two DataFrames using fuzzy string matching algorithms.

    This function performs entity resolution (also known as record linkage or deduplication) by
    comparing text fields between two DataFrames and identifying potential matches based on
    string similarity. It uses multiple fuzzy matching algorithms from the fuzzywuzzy library
    to compute match scores and can optionally filter candidates using Levenshtein distance.
    This is essential for linking records that refer to the same real-world entity despite
    variations in naming, spelling, or formatting.

    The function is particularly useful for:
      * Customer data deduplication across multiple systems
      * Merging company records from different databases
      * Matching product catalogs from various vendors
      * Linking person records with name variations
      * Data integration and master data management
      * Matching addresses with different formatting
      * Reconciling financial records across sources

    The function generates all possible pairs between the two DataFrames, calculates similarity
    scores using the specified fuzzy matching methods, and returns only pairs exceeding the
    match score threshold. Multiple matching algorithms can be used simultaneously to improve
    accuracy. Results include both the match scores and the original column values from both
    DataFrames for manual review.

    Parameters
    ----------
    dataframe_1
        First pandas DataFrame containing records to match. Must include the primary key
        column and the columns specified in columns_to_compare.
    dataframe_1_primary_key
        Name of the column in dataframe_1 that serves as the unique identifier for each
        record. This will be included in the output for reference.
    dataframe_2
        Second pandas DataFrame containing records to match against dataframe_1. Must
        include the primary key column and the columns specified in columns_to_compare.
    dataframe_2_primary_key
        Name of the column in dataframe_2 that serves as the unique identifier for each
        record. This will be included in the output for reference.
    levenshtein_distance_filter
        If specified, only pairs with Levenshtein edit distance less than or equal to
        this value are evaluated with fuzzy matching algorithms. This pre-filters candidates
        to improve performance. If None, no filtering is applied. Defaults to None.
    match_score_threshold
        Minimum similarity score (0-100) required for a pair to be included in results.
        Higher values are more restrictive and return only very similar matches. Typical
        values are 80-95 for fuzzy matching. Defaults to 95.
    columns_to_compare
        List of column names to use for entity matching. These columns are concatenated
        (with spaces) to create the comparison strings. If None, all common columns between
        the two DataFrames are used. Defaults to None.
    match_methods
        List of fuzzy matching algorithms to apply. Valid options are:
          * 'Ratio': Basic string similarity using difflib
          * 'Partial Ratio': Matches shortest string against substrings
          * 'Token Sort Ratio': Sorts tokens alphabetically before comparison
          * 'Partial Token Sort Ratio': Combination of partial and token sort
          * 'Token Set Ratio': Uses intersection of token sets
          * 'Partial Token Set Ratio': Combination of partial and token set
          * 'Weighted Ratio': Weighted average of multiple methods (recommended)
        Defaults to ['Partial Token Set Ratio', 'Weighted Ratio'].

    Returns
    -------
    pd.DataFrame
        A DataFrame containing potential matches with the following structure:
          * Primary key columns from both DataFrames (renamed with _Entity 1/_Entity 2)
          * 'Entity 1': Concatenated comparison string from dataframe_1
          * 'Entity 2': Concatenated comparison string from dataframe_2
          * One column per match method containing similarity scores (0-100)
          * Original columns from columns_to_compare (with _Entity 1/_Entity 2 suffixes)
        Each row represents a potential match between records from the two DataFrames.
        If no matches are found, returns an empty DataFrame with the correct structure.

    Examples
    --------
    # Match customer records from two different systems
    import pandas as pd
    customers_db1 = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'company_name': ['Acme Corporation', 'Tech Innovations Inc', 'Global Solutions Ltd'],
        'city': ['New York', 'Boston', 'Chicago']
    })
    customers_db2 = pd.DataFrame({
        'client_id': [101, 102, 103],
        'company_name': ['ACME Corp', 'Tech Innovations Incorporated', 'Global Solutns'],
        'city': ['New York', 'Boston', 'Chicago']
    })
    matches = ConductEntityMatching(
        customers_db1,
        'customer_id',
        customers_db2,
        'client_id',
        columns_to_compare=['company_name', 'city'],
        match_score_threshold=85
    )
    # Returns matches like: 'Acme Corporation' → 'ACME Corp' with high scores

    # Deduplicate product records with Levenshtein pre-filter
    products_current = pd.DataFrame({
        'sku': ['P001', 'P002', 'P003'],
        'product_name': ['Apple iPhone 13 Pro', 'Samsung Galaxy S21', 'Dell XPS 15']
    })
    products_incoming = pd.DataFrame({
        'new_sku': ['N001', 'N002', 'N003'],
        'product_name': ['Apple iPhone 13Pro', 'Samsung Galaxy S21 Ultra', 'Dell XPS Laptop']
    })
    matches = ConductEntityMatching(
        products_current,
        'sku',
        products_incoming,
        'new_sku',
        levenshtein_distance_filter=10,  # Pre-filter candidates
        match_score_threshold=80,
        columns_to_compare=['product_name'],
        match_methods=['Weighted Ratio', 'Token Set Ratio']
    )

    # Match person names with multiple matching methods
    employees = pd.DataFrame({
        'emp_id': [1, 2, 3],
        'full_name': ['John Smith', 'Mary Johnson', 'Robert Williams']
    })
    contractors = pd.DataFrame({
        'contractor_id': [10, 20, 30],
        'full_name': ['Smith, John', 'Mary A. Johnson', 'Bob Williams']
    })
    matches = ConductEntityMatching(
        employees,
        'emp_id',
        contractors,
        'contractor_id',
        columns_to_compare=['full_name'],
        match_score_threshold=90,
        match_methods=['Ratio', 'Partial Ratio', 'Token Sort Ratio', 'Weighted Ratio']
    )
    # Multiple methods help catch different name format variations

    """
    # Lazy load uncommon packages
    from fuzzywuzzy import fuzz
    import Levenshtein
    
    # Create list of valid match methods
    valid_match_methods = [
        'Ratio', 'Partial Ratio', 
        'Token Sort Ratio', 'Partial Token Sort Ratio', 
        'Token Set Ratio', 'Partial Token Set Ratio', 
        'Weighted Ratio'
    ]
    
    # Ensure that match methods are valid
    for match_method in match_methods:
        if match_method not in valid_match_methods:
            raise ValueError('Invalid match method specified: ' + match_method + ". Valid match methods are: " + ', '.join(valid_match_methods))
        
    # Ensure that lev distance threshold is either int or None
    if levenshtein_distance_filter is not None:
        if not isinstance(levenshtein_distance_filter, int):
            raise ValueError('Levenshtein distance threshold must be an integer or None')

    # Select columns to compare if specified
    if columns_to_compare is not None:
        dataframe_1 = dataframe_1[[dataframe_1_primary_key] + columns_to_compare].copy()
        dataframe_2 = dataframe_2[[dataframe_2_primary_key] + columns_to_compare].copy()
        
    # Remove duplicate rows from dataframe_1 and dataframe_2
    dataframe_1.drop_duplicates(inplace=True)
    dataframe_2.drop_duplicates(inplace=True)
    
    # Create placeholder dataframe to store results
    data_match_results = pd.DataFrame()
    
    # Add string column to dataframe_1 to store concatenated values of columns to compare
    if len(columns_to_compare) == 1:
        dataframe_1['Entity'] = dataframe_1[columns_to_compare[0]]
        dataframe_2['Entity'] = dataframe_2[columns_to_compare[0]]
    else:
        dataframe_1['Entity'] = dataframe_1[columns_to_compare[0]].str.cat(dataframe_1[columns_to_compare[1:]], sep=' ')
        dataframe_2['Entity'] = dataframe_2[columns_to_compare[0]].str.cat(dataframe_2[columns_to_compare[1:]], sep=' ')
    
    # Create combination of each Entity 1 and Entity 2
    entity_combinations = pd.merge(dataframe_1, dataframe_2, how='cross', suffixes=('_1', '_2'))
    
    # Calculate Levenshtein distance if specified
    if levenshtein_distance_filter is not None:
        entity_combinations['Levenshtein Distance'] = entity_combinations.apply(lambda row: Levenshtein.distance(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        entity_combinations = entity_combinations[entity_combinations['Levenshtein Distance'] <= levenshtein_distance_filter]
    
    # If dataframe_1_primary_key is not in entity_combinations, add _1 to the end of the column name
    if dataframe_1_primary_key not in entity_combinations.columns:
        dataframe_1_primary_key = dataframe_1_primary_key + '_1'
    
    # If dataframe_2_primary_key is not in entity_combinations, add _2 to the end of the column name
    if dataframe_2_primary_key not in entity_combinations.columns:
        dataframe_2_primary_key = dataframe_2_primary_key + '_2'
    
    # Create list of columns to select
    list_columns_to_select = [
        'Entity_1', 
        'Entity_2'
    ]
    
    # Calculate fuzz ratio, which calls difflib.ratio on the two input strings
    if 'Ratio' in match_methods:
        # Calculate fuzz ratio
        df_temp = entity_combinations[list_columns_to_select].copy()
        df_temp['Match Method'] = 'Ratio'
        df_temp['Match Score'] = entity_combinations.apply(lambda row: fuzz.ratio(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        df_temp = df_temp[df_temp['Match Score'] >= match_score_threshold]
        # Add the matches to the dataframe
        data_match_results = pd.concat([data_match_results, df_temp], ignore_index=True)
    
    # Calculate partial ratio, which calls ratio using the shortest string (length n) against all n-length substrings of the larger string and returns the highest score
    if 'Partial Ratio' in match_methods:
        # Calculate partial fuzz ratio
        df_temp = entity_combinations[list_columns_to_select].copy()
        df_temp['Match Method'] = 'Partial Ratio'
        df_temp['Match Score'] = entity_combinations.apply(lambda row: fuzz.partial_ratio(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        df_temp = df_temp[df_temp['Match Score'] >= match_score_threshold]
        # Add the matches to the dataframe
        data_match_results = pd.concat([data_match_results, df_temp], ignore_index=True)
        
    # Calculate token sort ratio, which tokenizes the strings and sorts them alphabetically before computing ratio
    if 'Token Sort Ratio' in match_methods:
        # Calculate token sort ratio
        df_temp = entity_combinations[list_columns_to_select].copy()
        df_temp['Match Method'] = 'Token Sort Ratio'
        df_temp['Match Score'] = entity_combinations.apply(lambda row: fuzz.token_sort_ratio(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        df_temp = df_temp[df_temp['Match Score'] >= match_score_threshold]
        # Add the matches to the dataframe
        data_match_results = pd.concat([data_match_results, df_temp], ignore_index=True)
        
    # Calculate partial token sort ratio, which tokenizes the strings and sorts them alphabetically before computing ratio using the shortest string (length n) against all n-length substrings of the larger string and returns the highest score
    if 'Partial Token Sort Ratio' in match_methods:
        # Calculate partial token sort ratio
        df_temp = entity_combinations[list_columns_to_select].copy()
        df_temp['Match Method'] = 'Partial Token Sort Ratio'
        df_temp['Match Score'] = entity_combinations.apply(lambda row: fuzz.partial_token_sort_ratio(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        df_temp = df_temp[df_temp['Match Score'] >= match_score_threshold]
        # Add the matches to the dataframe
        data_match_results = pd.concat([data_match_results, df_temp], ignore_index=True)
        
    # Calculate token set ratio, which tokenizes the strings and computes ratio using the intersection of the two token sets
    if 'Token Set Ratio' in match_methods:
        # Calculate token set ratio
        df_temp = entity_combinations[list_columns_to_select].copy()
        df_temp['Match Method'] = 'Token Set Ratio'
        df_temp['Match Score'] = entity_combinations.apply(lambda row: fuzz.token_set_ratio(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        df_temp = df_temp[df_temp['Match Score'] >= match_score_threshold]
        # Add the matches to the dataframe
        data_match_results = pd.concat([data_match_results, df_temp], ignore_index=True)
        
    # Calculate partial token set ratio, which tokenizes the strings and computes ratio using the intersection of the two token sets using the shortest string (length n) against all n-length substrings of the larger string and returns the highest score
    if 'Partial Token Set Ratio' in match_methods:
        # Calculate partial token set ratio
        df_temp = entity_combinations[list_columns_to_select].copy()
        df_temp['Match Method'] = 'Partial Token Set Ratio'
        df_temp['Match Score'] = entity_combinations.apply(lambda row: fuzz.partial_token_set_ratio(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        df_temp = df_temp[df_temp['Match Score'] >= match_score_threshold]
        # Add the matches to the dataframe
        data_match_results = pd.concat([data_match_results, df_temp], ignore_index=True)
    
    # Calculate weighted ratio, which is a weighted average of the above scores
    if 'Weighted Ratio' in match_methods:
        # Calculate weighted ratio
        df_temp = entity_combinations[list_columns_to_select].copy()
        df_temp['Match Method'] = 'Weighted Ratio'
        df_temp['Match Score'] = entity_combinations.apply(lambda row: fuzz.WRatio(row['Entity_1'], row['Entity_2']), axis=1)
        # Filter out combinations that are not within the threshold
        df_temp = df_temp[df_temp['Match Score'] >= match_score_threshold]
        # Add the matches to the dataframe
        data_match_results = pd.concat([data_match_results, df_temp], ignore_index=True)
    
    # Drop duplicates
    data_match_results.drop_duplicates(inplace=True)
    
    # If there are no matches, create an empty dataframe with the right structure
    if data_match_results.empty:
        # Create a dataframe with all entity combinations
        entity_pairs = entity_combinations[['Entity_1', 'Entity_2']].copy()
        
        # Create an empty DataFrame with the correct structure
        data_match_results = pd.DataFrame()
        
        # Add entity columns
        data_match_results['Entity 1'] = entity_pairs['Entity_1']
        data_match_results['Entity 2'] = entity_pairs['Entity_2']
        
        # Add empty columns for each match method
        for method in match_methods:
            data_match_results[method] = np.nan
    else:
        # Convert to wide-form dataframe
        data_match_results = data_match_results.pivot(index=['Entity_1', 'Entity_2'], columns='Match Method', values='Match Score')
        
        # Ensure all specified match methods are present as columns
        for method in match_methods:
            if method not in data_match_results.columns:
                data_match_results[method] = np.nan
        
        # Reset index of dataframe
        data_match_results['Entity 1'] = data_match_results.index.get_level_values(0)
        data_match_results['Entity 2'] = data_match_results.index.get_level_values(1)
        data_match_results.reset_index(drop=True, inplace=True)
    
    # Move entity columns to front
    data_match_results = data_match_results[['Entity 1', 'Entity 2'] + [col for col in data_match_results.columns if col not in ['Entity 1', 'Entity 2']]]
    
     # Reset column names
    data_match_results.columns.name = None
    
    # Left join columns to compare from dataframe 1
    if dataframe_1_primary_key not in dataframe_1.columns:
        dataframe_1_primary_key = dataframe_1_primary_key.replace('_1', '')
    data_match_results = data_match_results.merge(
        dataframe_1[[dataframe_1_primary_key, 'Entity'] + columns_to_compare], 
        how='left', 
        left_on='Entity 1', 
        right_on='Entity',
    )
    
    # Left join columns to compare from dataframe 2
    if dataframe_2_primary_key not in dataframe_2.columns:
        dataframe_2_primary_key = dataframe_2_primary_key.replace('_2', '')
    data_match_results = data_match_results.merge(
        dataframe_2[[dataframe_2_primary_key, 'Entity'] + columns_to_compare],
        how='left',
        left_on='Entity 2',
        right_on='Entity',
        suffixes=('_Entity 1', '_Entity 2')
    )
    
    # Drop entity columns
    data_match_results.drop(columns=['Entity_Entity 1', 'Entity_Entity 2'], inplace=True)
    
    # Move identifier columns to front
    if dataframe_1_primary_key not in data_match_results.columns:
        dataframe_1_primary_key = dataframe_1_primary_key + '_Entity 1'
    if dataframe_2_primary_key not in data_match_results.columns:
        dataframe_2_primary_key = dataframe_2_primary_key + '_Entity 2'
    data_match_results = data_match_results[[dataframe_1_primary_key, dataframe_2_primary_key] + [col for col in data_match_results.columns if col not in [dataframe_1_primary_key, dataframe_2_primary_key]]]
    
    # Drop duplicate matches
    data_match_results.drop_duplicates(inplace=True)
    
    # Return results
    return(data_match_results)

