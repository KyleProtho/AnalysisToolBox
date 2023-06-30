import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymetalog as pm
import seaborn as sns

def CreateMetalogDistribution(dataframe,
                              variable,
                              lower_bound=None,
                              upper_bound=None,
                              learning_rate=.01,
                              term_maximum=9,
                              term_minimum=2,
                              show_comparison_plot=True):
    """_summary_
    This function creates a metalog distribution from a dataset and variable.
    
    Args:
        dataframe (_type_): Pandas dataframe containing the dataset.
        variable (str): Name of the variable to create the metalog distribution from.
        lower_bound (int or float, optional): The lower bound of the metalog distribution. Defaults to None.
        upper_bound (int or float, optional): The upper bound of the metalog distribution. Defaults to None.
        learning_rate (float, optional): The learning rate to use when creating the metalog distribution. Defaults to .01.
        term_maximum (int, optional): The maximum number of terms to use when creating the metalog distribution. Defaults to 9.
        term_minimum (int, optional): The minimum number of terms to use when creating the metalog distribution. Defaults to 2.
        show_comparison_plot (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    # Select necessary columns from the dataframe
    dataframe = dataframe[[variable]]
    
    # Filter NA, None, and infinite values from the dataframe
    dataframe = dataframe[dataframe[variable].notna()]
    dataframe = dataframe[dataframe[variable] != None]
    dataframe = dataframe[dataframe[variable] != np.inf]
    
    # Extract values from the dataframe
    arr_variable = dataframe[variable].values
    
    # Create a metalog distribution
    if lower_bound is None and upper_bound is None:
        metalog_dist = pm.metalog(
            x=arr_variable,
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    elif lower_bound is not None and upper_bound is None:
        metalog_dist = pm.metalog(
            x=arr_variable,
            boundedness='sl',
            bounds=[lower_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    elif lower_bound is None and upper_bound is not None:
        metalog_dist = pm.metalog(
            x=arr_variable,
            boundedness='su',
            bounds=[upper_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
    else:
        metalog_dist = pm.metalog(
            x=arr_variable,
            boundedness='b',
            bounds=[lower_bound, upper_bound],
            step_len=learning_rate,
            term_lower_bound=term_minimum,
            term_limit=term_maximum
        )
        
    # Show summary of the metalog distribution
    pm.summary(m = metalog_dist)
    
    # To-Do: Add a comparison plot of the metalog distribution and the data
    
    # Return the metalog distribution
    return metalog_dist


# Test the function
from sklearn import datasets
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
iris['species'] = datasets.load_iris(as_frame=True).target
iris = iris[iris['species'] == 0]
# petal_legnth_dist = CreateMetalogDistribution(
#     dataframe=iris,
#     variable='petal length (cm)',
# )
# petal_legnth_dist = CreateMetalogDistribution(
#     dataframe=iris,
#     variable='petal length (cm)',
#     upper_bound=7,
# )
petal_legnth_dist = CreateMetalogDistribution(
    dataframe=iris,
    variable='petal length (cm)',
    lower_bound=0,
)
