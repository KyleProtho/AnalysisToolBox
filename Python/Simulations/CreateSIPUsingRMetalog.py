import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

def CreateSIPUsingRMetalog(dataframe,
                           variable_name,
                           lower_bound=None,
                           upper_bound=None,
                           trials=10000,
                           number_of_terms=4):
    # Import the CreateMetalogDistribution function from R
    robjects.r.source("https://raw.githubusercontent.com/onenonlykpro/SnippetsForStatistics/master/R/Simulations/CreateMetalogDistribution.R")
    
    # Import the SimulateOutcomeFromMetalog function from R
    robjects.r.source("https://raw.githubusercontent.com/onenonlykpro/SnippetsForStatistics/master/R/Simulations/SimulateOutcomeFromMetalog.R")
    
    # Select the variable from the dataframe
    dataframe = dataframe[[variable_name]]
    
    # Filter out missing values
    dataframe = dataframe.dropna()
    
    # Convert pandas dataframe to R dataframe
    pandas2ri.activate()
    dataframe = robjects.conversion.py2rpy(dataframe)
    
    # Create the metalog distribution
    if lower_bound == None and upper_bound == None:
        metalog_distribution = robjects.r.CreateMetalogDistribution(dataframe, variable_name, "u", )
    else:
        if lower_bound == None:
            metalog_distribution = robjects.r.CreateMetalogDistribution(dataframe, variable_name, "b", upper_bound)
        elif upper_bound == None:
            metalog_distribution = robjects.r.CreateMetalogDistribution(dataframe, variable_name, "b", robjects.NULL, lower_bound)
        else:
            metalog_distribution = robjects.r.CreateMetalogDistribution(dataframe, variable_name, "b", upper_bound, lower_bound)
            
    # Randomly sample from the metalog distribution
    data_simulated = robjects.r.SimulateOutcomeFromMetalog(metalog_distribution, number_of_terms, trials, variable_name)
    
    # Convert the R dataframe to a pandas dataframe
    data_simulated = robjects.conversion.rpy2py(data_simulated)
    
    # Return the simulated data
    return data_simulated


# Test the function
from sklearn.datasets import load_iris
iris = load_iris()
iris = pd.DataFrame(iris.data, columns=iris.feature_names)
# petal_length = CreateSIPUsingRMetalog(
#     dataframe=iris,
#     variable_name="petal length (cm)"
# )
# petal_length = CreateSIPUsingRMetalog(
#     dataframe=iris,
#     variable_name="petal length (cm)",
#     upper_bound=9
# )
# petal_length = CreateSIPUsingRMetalog(
#     dataframe=iris,
#     variable_name="petal length (cm)",
#     lower_bound=1
# )
petal_length = CreateSIPUsingRMetalog(
    dataframe=iris,
    variable_name="petal length (cm)",
    upper_bound=10,
    lower_bound=1
)
