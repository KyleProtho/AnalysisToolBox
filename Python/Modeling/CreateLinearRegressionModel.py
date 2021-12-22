import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm

def CreateLinearRegressionModel(dataframe,
                                outcome_variable,
                                list_of_predictors,
                                share_to_use_as_test_set = .20,
                                show_diagnostic_plots = True):
    # Select columns specified
    arr_columns = list_of_predictors.copy()
    arr_columns.append(outcome_variable)
    dataframe = dataframe[arr_columns]
    
    # Shuffle the rows of dataframe
    dataframe = dataframe.sample(frac = 1)
    
    # Create predictor/independent variables
    X = dataframe[list_of_predictors]
    X = sm.add_constant(X)
    
    # Create outcome/dependent variables
    Y = dataframe[outcome_variable]
    
    # Determine size of test dataset
    test_size = round(len(dataframe.index) * share_to_use_as_test_set, 0)
    test_size = int(test_size)
    
    # Split the data into training/testing sets
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    
    # Split the outcomes into training/testing sets
    Y_train = Y[:-test_size]
    Y_test = Y[-test_size:]
    
    # Create linear regression model
    model = sm.OLS(Y_train, X_train)
    model_res = model.fit()
    
    # If specified, show diagnostic plots
    if show_diagnostic_plots:
        for variable in list_of_predictors:
            fig = plt.figure(figsize=(12, 8))
            fig = sm.graphics.plot_regress_exog(model_res, variable, fig=fig)
    
    # Create dictionary of objects to return
    dict_return = {
        "Fitted Model": model_res,
        "Predictor Test Dataset": X_test,
        "Outcome Test Dataset": Y_test
    }
    return dict_return
