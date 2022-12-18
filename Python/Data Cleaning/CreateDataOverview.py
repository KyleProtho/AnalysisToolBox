import pandas as pd

def CreateDataOverview(dataframe,
                       return_data_overview=False):
    # Get data types in each column
    data_overview = pd.DataFrame(dataframe.dtypes, columns=['DataType'])
    data_overview = data_overview.reset_index()
    data_overview = data_overview.rename(columns={
        'index': 'Variable',
        'DataType': 'Data Type'
    })
    
    # Show missing values and unique value counts in each column
    data_summary = dataframe.describe(include='all').T
    data_summary = data_summary.reset_index()
    data_summary = data_summary.rename(columns={
        'index': 'Variable'
    })
    try:
        data_summary = data_summary[['Variable', 'count', 'unique', 'top', 'freq', 'min', 'max']]
    except KeyError:
        data_summary = data_summary[['Variable', 'count', 'min', 'max']]
    
    # Join the two dataframes
    data_overview = data_overview.merge(
        data_summary, 
        how='left',
        on='Variable')
    del(data_summary)
    
    # Print the overview
    print(data_overview)
    
    # Return the overview, if requested
    if return_data_overview:
        return(data_overview)


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# # iris['species'] = iris['species'].astype('category')
# CreateDataOverview(iris)
