# Load packages
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd

# Declare function
def CreateBinnedColumn(dataframe,
                       numeric_column_name,
                       number_of_bins=6,
                       binning_strategy='kmeans',
                       new_column_name=None):
    """
    Discretize continuous numeric data into discrete bins using specified strategies.

    This function transforms a continuous variable into a categorical one by grouping
    values into bins. It utilizes scikit-learn's `KBinsDiscretizer` to perform the
    binning, supporting various strategies like equal width, equal frequency (quantiles), or
    k-means clustering. This is essential for converting numeric features into features 
    suitable for algorithms that require categorical input or for simplifying 
    complex distributions for analysis.

    The function is particularly useful for:
      * Feature engineering for machine learning models (e.g., decision trees, naïve bayes)
      * Segmenting customers into groups (e.g., age groups, income brackets)
      * Reducing noise and handling outliers in continuous data
      * Creating histograms or grouped summaries for exploratory data analysis
      * Simplifying the interpretation of complex numeric distributions
      * Identifying natural clusters within a single dimension using k-means

    The function handles missing and non-finite values by excluding them from the binning 
    calculation and returning them as NaN in the resulting column.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing high-cardinality numeric data.
    numeric_column_name
        The name of the column containing the continuous numeric values to be binned.
    number_of_bins
        The number of bins to create. Defaults to 6.
    binning_strategy
        The strategy used to define the widths of the bins. Options include:
          * 'uniform': All bins in each feature have identical widths.
          * 'quantile': All bins in each feature have the same number of points.
          * 'kmeans': Values in each bin have the same nearest center of a 1D k-means cluster.
        Defaults to 'kmeans'.
    new_column_name
        The name of the new column to be added to the DataFrame. If None, the column 
        will be named '{numeric_column_name}- Binned'. Defaults to None.

    Returns
    -------
    pd.DataFrame
        An updated pandas DataFrame containing the original columns plus a new column 
        with ordinal bin identifiers (labeled 0, 1, 2, ...).

    Examples
    --------
    # Create age groups using k-means clustering
    import pandas as pd
    demographics = pd.DataFrame({
        'user_id': range(1, 7),
        'age': [18, 22, 35, 42, 58, 65]
    })
    demographics = CreateBinnedColumn(
        demographics, 
        numeric_column_name='age', 
        number_of_bins=3
    )
    # Adds 'age- Binned' column with values 0, 1, or 2

    # Segment income into equal-width bins
    financial_data = pd.DataFrame({
        'income': [20000, 45000, 50000, 120000, 250000, 80000]
    })
    financial_data = CreateBinnedColumn(
        financial_data, 
        numeric_column_name='income', 
        binning_strategy='uniform',
        number_of_bins=5,
        new_column_name='income_segment'
    )

    # Use quantiles to ensure equal distribution of users across bins
    web_traffic = pd.DataFrame({
        'session_duration': [10, 12, 11, 100, 200, 150, 15, 18]
    })
    web_traffic = CreateBinnedColumn(
        web_traffic, 
        numeric_column_name='session_duration', 
        binning_strategy='quantile',
        number_of_bins=4
    )

    """
    
    # Create new column name if not provided
    if new_column_name is None:
        new_column_name = numeric_column_name + '- Binned'
        
    # Select only the variable to bin
    dataframe_bin = dataframe[[numeric_column_name]].copy()
    
    # Keep complete cases only
    dataframe_bin = dataframe_bin.dropna()
    
    # Keep only finite values
    dataframe_bin = dataframe_bin[np.isfinite(dataframe_bin).all(1)]
        
    # Create the array to bin
    array_to_bin = np.array(dataframe_bin[numeric_column_name]).reshape(-1, 1)
    
    # Create the discretizer
    binner = KBinsDiscretizer(n_bins=number_of_bins, 
                              strategy=binning_strategy,
                              encode='ordinal')
    
    # Fit the discretizer
    binner.fit(array_to_bin)
    
    # Create the binned variable
    dataframe_bin[new_column_name] = binner.transform(array_to_bin)
    
    # Merge the binned variable back into the original dataframe
    dataframe = dataframe.merge(
        dataframe_bin[[new_column_name]],
        how='left',
        right_index=True,
        left_index=True
    )
    
    # Return the dataframe
    return dataframe

