# Load packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

# Declare function
def CreateDataOverview(dataframe,
                       plot_missingness=False):
    """
    Generate a comprehensive technical summary and data dictionary for a DataFrame.

    This function analyzes a pandas DataFrame to create an overview of its structure,
    content, and quality. It calculates essential metrics for each variable, including
    data types, missing value counts, missing percentages, and summary statistics
    such as unique value counts, top values, frequency, minimums, maximums, and ranges.
    This consolidated view is essential for understanding the data's composition
    before performing more advanced analysis.

    The function is particularly useful for:
      * Creating automated data dictionaries for documentation
      * Auditing data completeness and identifying problematic columns
      * Comparing data types against expected schemas
      * Rapidly assessing the distribution and boundaries of numeric features
      * Identifying high-cardinality categorical variables
      * Visualizing missingness patterns across large datasets
      * Initial Exploratory Data Analysis (EDA) on new data sources

    When `plot_missingness` is enabled, the function generates a sorted bar chart
    showing the percentage of missing values for each column, allowing for quick
    visual identification of data gaps.

    Parameters
    ----------
    dataframe
        The pandas DataFrame to analyze. Can contain numeric, categorical, and
        datetime data types.
    plot_missingness
        If True, generates a horizontal bar chart visualizing the percentage of
        missing values for each column, sorted by descending missingness.
        Defaults to False.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame where each row enters a variable from the original dataset.
        Columns include:
          * Variable: Name of the column
          * Data Type: Pandas dtype (e.g., int64, float64, object)
          * Missing Count: Number of NaN or null values
          * Missing Percentage: Ratio of missing values to total rows
          * Non Missing Count: Count of valid observations
          * Unique Value Count: Number of distinct values
          * Top Value: Most frequent value (for categorical or mixed data)
          * Frequency of Top Value: Count of the most frequent value
          * Minimum: Minimum value in the series
          * Maximum: Maximum value in the series
          * Range: Difference between Maximum and Minimum (for numeric data)

    Examples
    --------
    # Generate a basic data overview
    import pandas as pd
    import numpy as np
    sales_data = pd.DataFrame({
        'TransactionID': range(100),
        'Product': ['A', 'B', 'C', np.nan] * 25,
        'Amount': np.random.uniform(10, 500, 100),
        'Date': pd.date_range('2023-01-01', periods=100)
    })
    overview = CreateDataOverview(sales_data)
    # Returns summary statistics for all 4 columns

    # View missingness patterns with a plot
    customer_survey = pd.DataFrame({
        'UserID': range(50),
        'Satisfaction': [1, 5, 3] * 16 + [np.nan, np.nan],
        'Feedback': [np.nan] * 40 + ['Good'] * 10
    })
    overview = CreateDataOverview(customer_survey, plot_missingness=True)
    # Displays a plot showing UserID (0%), Satisfaction (4%), and Feedback (80%) missingness

    """
    
    # Get data types in each column
    data_overview = pd.DataFrame(dataframe.dtypes, columns=['DataType'])
    data_overview = data_overview.reset_index()
    data_overview = data_overview.rename(columns={
        'index': 'Variable',
        'DataType': 'Data Type'
    })
    
    # Count missing values in each column
    data_missing = dataframe.isnull().sum().reset_index()
    data_missing = data_missing.reset_index()
    data_missing = data_missing.rename(columns={
        'index': 'Variable',
        0: 'Missing Count'
    })
    data_missing = data_missing[['Variable', 'Missing Count']]
    
    # Join missing count to the overview
    data_overview = data_overview.merge(
        data_missing,
        how='left',
        on='Variable'
    )
    del(data_missing)
    
    # Calculate missing percentage
    data_overview['Missing Percentage'] = data_overview['Missing Count'] / len(dataframe)
    
    # Show range and frequency in each column
    data_summary = dataframe.describe(include='all').T
    data_summary = data_summary.reset_index()
    data_summary.columns.values[0] = 'Variable'
    try:
        data_summary = data_summary[['Variable', 'count', 'unique', 'top', 'freq', 'min', 'max']]
        data_summary = data_summary.rename(columns={
            'count': 'Non Missing Count',
            'unique': 'Unique Value Count',
            'top': 'Top Value',
            'freq': 'Frequency of Top Value',
            'min': 'Minimum',
            'max': 'Maximum'
        })
    except KeyError:
        try:
            data_summary = data_summary[['Variable', 'count', 'min', 'max']]
            data_summary = data_summary.rename(columns={
                'count': 'Non Missing Count',
                'min': 'Minimum',
                'max': 'Maximum'
            })
        except KeyError:
            data_summary = data_summary[['Variable', 'count', 'unique', 'top', 'freq']]
            data_summary = data_summary.rename(columns={
                'count': 'Non Missing Count',
                'unique': 'Unique Value Count',
                'top': 'Top Value',
                'freq': 'Frequency of Top Value'
            })
    
    # Join the two dataframes
    data_overview = data_overview.merge(
        data_summary, 
        how='left',
        on='Variable')
    del(data_summary)
    
    # Calculate the range of each column
    try:
        data_overview['Range'] = data_overview['Maximum'] - data_overview['Minimum']
    except KeyError:
        pass
    
    # Generate missingness plot, if requested
    if plot_missingness:
        # Sort the data by missing percentage
        data_overview_sorted = data_overview.sort_values(
            by=['Missing Percentage', 'Variable'],
            ascending=[False, True]
        )
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, len(data_overview_sorted.index)))
        sns.barplot(
            data=data_overview_sorted, 
            y='Variable', 
            x='Missing Percentage',
            palette='Blues_d'
        )
        # Add data labels
        lbls = [f"{x:.0%}" for x in data_overview_sorted['Missing Percentage']]
        ax.bar_label(container=ax.containers[0],
                     labels=lbls,
                     padding=5)
        # Set y-axis limits and hide ticks
        plt.xlim(0, 1)
        plt.xticks([])
        # Hide the spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # Format the x-axis labels and wrap them
        plt.gca().yaxis.set_tick_params(labelsize=8)
        y_labels = plt.gca().get_yticklabels()
        wrapped_labels = []
        for label in y_labels:
            wrap_label = '\n'.join(wrap(label.get_text(), 30))
            wrapped_labels.append(wrap_label)
        plt.gca().set_yticklabels(wrapped_labels)
        # Show the plot
        plt.show()
    
    # Return the overview
    return(data_overview)

