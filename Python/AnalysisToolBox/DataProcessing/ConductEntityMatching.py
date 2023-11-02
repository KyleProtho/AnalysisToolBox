# Load packages
from fuzzywuzzy import fuzz
import Levenshtein
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
    
    # Convert to wide-form dataframe
    data_match_results = data_match_results.pivot(index=['Entity_1', 'Entity_2'], columns='Match Method', values='Match Score')
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


# # Test function
# # Import dataset
# dataframe = pd.read_csv("C:/Users/oneno/Downloads/patients.csv")
# # Remove numbers from names
# dataframe['FIRST'] = dataframe['FIRST'].str.replace('\d+', '', regex=True)
# dataframe['LAST'] = dataframe['LAST'].str.replace('\d+', '', regex=True)
# # Split dataframe into two
# dataframe_1 = dataframe.iloc[:int(len(dataframe)/2)]
# dataframe_2 = dataframe.iloc[int(len(dataframe)/2):]
# # Perform entity matching
# # matches = ConductEntityMatching(
# #     dataframe_1, 'Id', 
# #     dataframe_2, 'Id', 
# #     columns_to_compare=['FIRST', 'LAST'],
# #     match_score_threshold=86,
# #     match_methods=['Weighted Ratio', 'Token Sort Ratio']
# # )
# matches = ConductEntityMatching(
#     dataframe_1, 'Id', 
#     dataframe_2, 'Id', 
#     columns_to_compare=['FIRST', 'LAST'],
#     levenshtein_distance_filter=10,
#     match_score_threshold=86,
#     match_methods=['Weighted Ratio', 'Token Sort Ratio']
# )
