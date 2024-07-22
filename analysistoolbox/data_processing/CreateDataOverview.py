# Load packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

# Declare function
def CreateDataOverview(dataframe,
                       plot_missingness=False):
    """
    This function creates an overview of the data in a dataframe, showing the data type, missing count, missing percentage, and summary statistics for each variable.
    Tip: This function is useful for creating a data dictionary.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        plot_missingness (bool, optional): Generates a plot to show missingness in each variable. Defaults to False.

    Returns:
        Pandas dataframe: A Pandas dataframe containing an overview of the data in the dataframe.
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

