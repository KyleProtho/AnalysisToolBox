# Load packages
from fuzzywuzzy import fuzz
import pandas as pd

# Declare function
def ConductEntityMatching(dataframe_1, 
                          dataframe_1_primary_key, 
                          dataframe_2, 
                          dataframe_2_primary_key, 
                          match_score_threshold=90,
                          columns_to_compare=None,
                          match_methods=['Partial Token Set Ratio', 'Weighted Ratio'],
                          show_progress=True):
    """
    Perform entity matching between two dataframes.
    
    :param dataframe_1: First dataframe
    :param dataframe_2: Second dataframe
    :param dataframe_1_primary_key: Key column in first dataframe
    :param dataframe_2_primary_key: Key column in second dataframe
    :param match_score_threshold: match_score_threshold for cosine similarity (default=0.8)
    :return: Dataframe with matched entities
    """
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

    # Select columns to compare if specified
    if columns_to_compare is not None:
        dataframe_1 = dataframe_1[[dataframe_1_primary_key] + columns_to_compare].copy()
        dataframe_2 = dataframe_2[[dataframe_2_primary_key] + columns_to_compare].copy()
    
    # Create placeholder dataframe to store results
    data_match_results = pd.DataFrame()
    
    # Iterate through each row in dataframe_1
    for index_1, row_1 in dataframe_1.iterrows():
        # Show progress based on number of rows
        if show_progress:
            if index_1 + 1 % 50 == 0:
                print("Processing row " + str(index_1 + 1) + " of " + str(len(dataframe_1)))
        
        # Get the identifier of the row
        entity_1_id = row_1[dataframe_1_primary_key]
        # Create string to store concatenated values of columns to compare
        entity_1 = row_1[columns_to_compare].str.cat(sep=' ')
        
        # Iterate through each row in dataframe_2
        for index_2, row_2 in dataframe_2.iterrows():
            # Get the identifier of the row
            entity_2_id = row_2[dataframe_2_primary_key]
            # Create string to store concatenated values of columns to compare
            entity_2 = row_2[columns_to_compare].str.cat(sep=' ')
            
            # Calculate fuzz ratio, which calls difflib.ratio on the two input strings
            if 'Ratio' in match_methods:
                fuzz_ratio_score = fuzz.ratio(entity_1, entity_2)
                # If score is above the threshold, then add the match to the dataframe
                if fuzz_ratio_score >= match_score_threshold:
                    df = pd.DataFrame({
                        'Entity 1': entity_1_id, 'Entity 2': entity_2_id, 
                        'Match Method': "Ratio", 'Match Score': fuzz_ratio_score
                    }, index=[0])
                    data_match_results = pd.concat([data_match_results, df], ignore_index=True)
            
            # Calculate partial ratio, which calls ratio using the shortest string (length n) against all n-length substrings of the larger string and returns the highest score
            if 'Partial Ratio' in match_methods:
                partial_ratio_score = fuzz.partial_ratio(entity_1, entity_2)
                # If score is above the threshold, then add the match to the dataframe
                if partial_ratio_score >= match_score_threshold:
                    df = pd.DataFrame({
                        'Entity 1': entity_1_id, 'Entity 2': entity_2_id, 
                        'Match Method': "Patial Ratio", 'Match Score': partial_ratio_score
                    }, index=[0])
                    data_match_results = pd.concat([data_match_results, df], ignore_index=True)
                
            # Calculate token sort ratio, which tokenizes the strings and sorts them alphabetically before computing ratio
            if 'Token Sort Ratio' in match_methods:
                token_sort_ratio_score = fuzz.token_sort_ratio(entity_1, entity_2)
                # If score is above the threshold, then add the match to the dataframe
                df = pd.DataFrame({
                        'Entity 1': entity_1_id, 'Entity 2': entity_2_id, 
                        'Match Method': "Token Sort Ratio", 'Match Score': token_sort_ratio_score
                    }, index=[0])
                if token_sort_ratio_score >= match_score_threshold:
                    data_match_results = pd.concat([data_match_results, df], ignore_index=True)
            
            # Calculate partial token sort ratio, which tokenizes the strings and sorts them alphabetically before computing ratio using the shortest string (length n) against all n-length substrings of the larger string and returns the highest score
            if 'Partial Token Sort Ratio' in match_methods:
                partial_token_sort_ratio_score = fuzz.partial_token_sort_ratio(entity_1, entity_2)
                # If score is above the threshold, then add the match to the dataframe
                df = pd.DataFrame({
                        'Entity 1': entity_1_id, 'Entity 2': entity_2_id, 
                        'Match Method': "Partial Token Sort Ratio", 'Match Score': partial_token_sort_ratio_score
                    }, index=[0])
                if partial_token_sort_ratio_score >= match_score_threshold:
                    data_match_results = pd.concat([data_match_results, df], ignore_index=True)
                
            # Calculate token set ratio, which tokenizes the strings and computes ratio using the intersection of the two token sets
            if 'Token Set Ratio' in match_methods:
                token_set_ratio_score = fuzz.token_set_ratio(entity_1, entity_2)
                # If score is above the threshold, then add the match to the dataframe
                if token_set_ratio_score >= match_score_threshold:
                    df = pd.DataFrame({
                        'Entity 1': entity_1_id, 'Entity 2': entity_2_id, 
                        'Match Method': "Token Set Ratio", 'Match Score': token_set_ratio_score
                    }, index=[0])
                    data_match_results = pd.concat([data_match_results, df], ignore_index=True)
                
            # Calculate partial token set ratio, which tokenizes the strings and computes ratio using the intersection of the two token sets using the shortest string (length n) against all n-length substrings of the larger string and returns the highest score
            if 'Partial Token Set Ratio' in match_methods:
                partial_token_set_ratio_score = fuzz.partial_token_set_ratio(entity_1, entity_2)
                # If score is above the threshold, then add the match to the dataframe
                if partial_token_set_ratio_score >= match_score_threshold:
                    df = pd.DataFrame({
                        'Entity 1': entity_1_id, 'Entity 2': entity_2_id, 
                        'Match Method': "Partial Token Set Ratio", 'Match Score': partial_token_set_ratio_score
                    }, index=[0])
                    data_match_results = pd.concat([data_match_results, df], ignore_index=True)
            
            # Calculate weighted ratio, which is a weighted average of the above scores
            if 'Weighted Ratio' in match_methods:
                weighted_ratio_score = fuzz.WRatio(entity_1, entity_2)
                # If score is above the threshold, then add the match to the dataframe
                df = pd.DataFrame({
                        'Entity 1': entity_1_id, 'Entity 2': entity_2_id, 
                        'Match Method': "Weighted Ratio", 'Match Score': weighted_ratio_score
                    }, index=[0])
                if weighted_ratio_score >= match_score_threshold:
                    data_match_results = pd.concat([data_match_results, df], ignore_index=True)
    
    # Convert to wide-form dataframe
    data_match_results = data_match_results.pivot(index=['Entity 1', 'Entity 2'], columns='Match Method', values='Match Score')
    
    # # Reset index of dataframe
    data_match_results['Entity 1'] = data_match_results.index.get_level_values(0)
    data_match_results['Entity 2'] = data_match_results.index.get_level_values(1)
    data_match_results.reset_index(drop=True, inplace=True)
    
    # Move entity columns to front
    data_match_results = data_match_results[['Entity 1', 'Entity 2'] + [col for col in data_match_results.columns if col not in ['Entity 1', 'Entity 2']]]
    
     # Reset column names
    data_match_results.columns.name = None
    
    # Left join columns to compare from dataframe 1
    data_match_results = data_match_results.merge(
        dataframe_1[[dataframe_1_primary_key] + columns_to_compare], 
        how='left', 
        left_on='Entity 1', 
        right_on=dataframe_1_primary_key
    )
    
    # Left join columns to compare from dataframe 2
    data_match_results = data_match_results.merge(
        dataframe_2[[dataframe_2_primary_key] + columns_to_compare],
        how='left',
        left_on='Entity 2',
        right_on=dataframe_2_primary_key,
        suffixes=('_Entity 1', '_Entity 2')
    )
    
    # Return results
    return(data_match_results)


# # Test function
# # Import dataset
# dataframe = pd.read_csv("C:/Users/oneno/Downloads/patients.csv")
# # Remove numbers from names
# dataframe['FIRST'] = dataframe['FIRST'].str.replace('\d+', '', regex=True)
# dataframe['LAST'] = dataframe['LAST'].str.replace('\d+', '', regex=True)
# # Split dataframe into two
# dataframe_1 = dataframe.iloc[:int(len(dataframe)/3)]
# dataframe_2 = dataframe.iloc[int(len(dataframe)/3):]
# # Perform entity matching
# matches = ConductEntityMatching(
#     dataframe_1, 'Id', 
#     dataframe_2, 'Id', 
#     columns_to_compare=['FIRST', 'LAST'],
#     match_score_threshold=86,
#     match_methods=['Weighted Ratio', 'Token Sort Ratio']
# )
