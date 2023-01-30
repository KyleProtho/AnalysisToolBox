# Load packages
import pandas as pd
import numpy as np
import seaborn as sns
from fitter import Fitter, get_common_distributions

def IdentityDistribution(dataframe,
                         quantitative_variable_column_name):
    """_summary_
    This function uses the Fitter package to estimate the distribution of a quantitative variable.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        quantitative_variable_column_name (str): The name of the column containing the quantitative variable.

    Returns:
        Fitter: Fitter object containing the results of the distribution fitting.
    """
    
    # Convert column to numpy array
    df_no_nulls = dataframe[
        (dataframe[quantitative_variable_column_name].notnull())
    ]
    array_of_values = df_no_nulls[quantitative_variable_column_name].values
    
    # Use Fitter package to estimate distribution
    f = Fitter(array_of_values,
               distributions=get_common_distributions())
    f.fit()
    results = f.summary()
    
    # Return fitting summary
    return results


# # Test
# # Import kaggle dataset
# df_products = pd.read_excel("C:/Users/oneno/OneDrive/Data/Risk Modeling - Practice Datasets/Predicting Count of Products Sold.xlsx",
#                             sheet_name="Train Final")
# # Test function
# IdentityDistribution(dataframe=df_products,
#                      quantitative_variable_column_name='Count Sold')

