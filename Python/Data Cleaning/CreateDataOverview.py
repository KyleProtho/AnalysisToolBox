import pandas as pd

def CreateDataOverview(dataframe):
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
    data_summary = data_summary.rename(columns={
        'index': 'Variable'
    })
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
        data_summary = data_summary[['Variable', 'count', 'min', 'max']]
        data_summary = data_summary.rename(columns={
            'count': 'Non Missing Count',
            'min': 'Minimum',
            'max': 'Maximum'
        })
    
    # Join the two dataframes
    data_overview = data_overview.merge(
        data_summary, 
        how='left',
        on='Variable')
    del(data_summary)
    
    # Return the overview
    return(data_overview)


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# # iris['species'] = iris['species'].astype('category')
# CreateDataOverview(iris)
