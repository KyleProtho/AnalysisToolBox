# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Declare function
def ConductLinearRegressionAnalysis(dataframe,
                                    outcome_variable,
                                    list_of_predictors,
                                    add_constant=False,
                                    scale_predictors=False,
                                    show_diagnostic_plots_for_each_predictor=False,
                                    show_help=False):
    """
    Conducts a linear regression analysis on a given dataframe, using a specified outcome variable and list of predictor variables.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data to be analyzed.
        outcome_variable (str): The name of the column in the dataframe containing the outcome variable.
        list_of_predictors (list): A list of strings, each containing the name of a column in the dataframe containing a predictor variable.
        scale_predictors (bool, optional): Whether or not to scale the predictor variables before running the analysis. Defaults to False.
        show_diagnostic_plots_for_each_predictor (bool, optional): Whether or not to show diagnostic plots for each predictor variable. Defaults to False.
        show_help (bool, optional): Whether or not to show help text explaining how to access the output of the function. Defaults to False.

    Returns:
        dict: A dictionary containing the fitted linear regression model, the model summary, a test dataset of predictors, and a test dataset of outcomes.
    """
    
    # Select columns specified
    dataframe = dataframe[list_of_predictors + [outcome_variable]]
    
    # Remove NAN and inf values
    dataframe.dropna(inplace=True)
    dataframe = dataframe[np.isfinite(dataframe).all(1)]

    # Add constant
    if add_constant:
        dataframe = sm.add_constant(dataframe)
    
    # Scale the predictors, if requested
    if scale_predictors:
        # Show the mean and standard deviation of each predictor
        print("\nMean of each predictor:")
        print(dataframe[list_of_predictors].mean())
        print("\nStandard deviation of each predictor:")
        print(dataframe[list_of_predictors].std())
        
        # Scale predictors
        dataframe[list_of_predictors] = StandardScaler().fit_transform(dataframe[list_of_predictors])
    
    # Create linear regression model
    if add_constant:
        model = sm.OLS(dataframe[outcome_variable], dataframe[['const'] + list_of_predictors])
    else:
        model = sm.OLS(dataframe[outcome_variable], dataframe[list_of_predictors])
    model_res = model.fit()
    model_summary = model_res.summary()
    
    # If requested, show diagnostic plots
    if show_diagnostic_plots_for_each_predictor:
        for variable in list_of_predictors:
            fig = plt.figure(figsize=(12, 8))
            fig = sm.graphics.plot_regress_exog(model_res, variable, fig=fig)
    
    # If requested, show help text
    if show_help:
        print(
            "Quick guide on accessing output of ConductLinearRegressionAnalysis function:",
            "\nThe ouput of the ConductLinearRegressionAnalysis function is a dictionary containing the regression results, a test dataset of predictors, and a test dataset of outcomes.",
            "\n\t--To access the linear regression model, use the 'Fitted Model' key."
            "\n\t--To view the model's statistical summary, use the 'Model Summary' key."
        )
    
    # Create dictionary of objects to return
    dict_return = {
        "Fitted Model": model_res,
        "Model Summary": model_summary
    }
    return dict_return

