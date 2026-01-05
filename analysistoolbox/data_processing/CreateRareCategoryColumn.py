# Load packages 
import numpy as np
import pandas as pd

# Declare function
def CreateRareCategoryColumn(dataframe,
                             categorical_column_name,
                             rare_category_label="Other",
                             rare_category_threshold=0.01,
                             new_column_suffix=None):
    """
    Consolidate low-frequency categorical values into a single catch-all category.

    This function identifies categories within a specified column that appear less 
    frequently than a given threshold (percentage of total rows) and replaces them 
    with a single label (e.g., "Other"). This is a common preprocessing step to 
    reduce dimensionality and improve the stability of statistical models.

    The function is particularly useful for:
      * Reducing high cardinality in categorical features
      * Improving the performance and stability of machine learning models
      * Simplifying visualizations by grouping long tails of rare categories
      * Handling sparse categories that lack statistical significance
      * Data cleaning where numerous variations of minor labels exist
      * Standardizing categorical data for reporting and dashboards

    By default, the function creates a new column to preserve the original data, 
    appending a descriptive suffix to the original column name.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing categorical data.
    categorical_column_name
        The name of the column containing the categorical variable to analyze. 
        Supports string or object dtypes.
    rare_category_label
        The string label to assign to any category that falls below the threshold.
        Defaults to "Other".
    rare_category_threshold
        The minimum relative frequency (as a float between 0 and 1) required for 
        a category to remain distinct. Categories with a frequency lower than 
        this value will be grouped. Defaults to 0.01 (1%).
    new_column_suffix
        The suffix to append to the original column name for the new binned 
        column. If None, it defaults to " (with {rare_category_label})".
        Defaults to None.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with a new column added, where infrequent categories 
        have been replaced by the `rare_category_label`.

    Examples
    --------
    # Group rare industries into "Other" (threshold 15%)
    import pandas as pd
    clients = pd.DataFrame({
        'company': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'industry': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Retail', 'Agri', 'Hosp', 'Edu', 'Gov']
    })
    clients = CreateRareCategoryColumn(
        clients, 
        'industry', 
        rare_category_threshold=0.15
    )
    # 'Agri', 'Hosp', 'Edu', 'Gov' each appear only 10% and will be grouped into 'Other'

    # Use a custom label and specific suffix
    cities = pd.DataFrame({
        'city': ['NYC', 'NYC', 'NYC', 'LA', 'LA', 'SF', 'CHI', 'MIA']
    })
    cities = CreateRareCategoryColumn(
        cities, 
        'city', 
        rare_category_label='Minor City', 
        rare_category_threshold=0.20,
        new_column_suffix='_binned'
    )
    # NYC (37.5%) and LA (25%) stay; SF, CHI, MIA (12.5% each) become 'Minor City'

    """
    
    # If new column name is not provided, create one
    if new_column_suffix is None:
        new_column_suffix = " (with " + rare_category_label + ")"
    
    # Get relative frequency of each category
    data_value_relative_frequency = dataframe[categorical_column_name].value_counts(normalize=True)
    data_value_relative_frequency = pd.DataFrame(data_value_relative_frequency)
    data_value_relative_frequency.reset_index(inplace=True)
    
    # Add flag for rare categories
    data_value_relative_frequency['rare'] = data_value_relative_frequency['proportion'] < rare_category_threshold
    
    # Create new column name
    new_column_name = categorical_column_name + new_column_suffix
    
    # Get list of rare categories
    rare_values = data_value_relative_frequency[data_value_relative_frequency['rare'] == True][categorical_column_name].tolist()
    
    # Create new column in original dataframe with rare categories
    dataframe[new_column_name] = np.where(
        dataframe[categorical_column_name].isnull(), np.nan,
        np.where(dataframe[categorical_column_name].isin(rare_values), rare_category_label, 
        dataframe[categorical_column_name])
    )
    
    # Return the dataframe
    return dataframe

