import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

def PlotCorrelationMatrix(dataframe,
                          list_of_columns,
                          show_as_pairplot = True):
    # Select relevant variables, keep complete cases only
    completed_df = dataframe[list_of_columns].dropna()
    
    # Drop Inf values
    completed_df = completed_df[np.isfinite(completed_df).all(1)]
    print("Count of complete observations for correlation matrix: " + str(len(completed_df.index)))
    
    # Show pairplot if specified -- otherwise, print correlation matrix
    if show_as_pairplot:
        sns.pairplot(completed_df)
    else:
        print(completed_df.corr())
        
    # Return complete case dataset
    return(completed_df)
