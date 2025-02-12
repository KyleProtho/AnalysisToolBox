# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Declare function
def ConductLogisticRegressionAnalysis(dataframe,
                                      outcome_variable,
                                      list_of_predictors,
                                      add_constant=True,
                                      scale_predictors=False,
                                      show_diagnostic_plots_for_each_predictor=True,
                                      show_help=False):
    """
    Conducts a logistic regression analysis on the specified dataframe, using the specified outcome variable and list of predictors.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data to be analyzed.
        outcome_variable (str): The name of the column in the dataframe containing the outcome variable.
        list_of_predictors (list of str): A list of the names of the columns in the dataframe containing the predictor variables.
        scale_predictors (bool, optional): Whether or not to scale the predictor variables. Default is False.
        show_diagnostic_plots_for_each_predictor (bool, optional): Whether or not to show diagnostic plots for each predictor variable. Default is True.
        show_help (bool, optional): Whether or not to show help text. Default is False.

    Returns:
        dict: A dictionary containing the fitted model and model summary.
    """
    
    # Select columns specified
    dataframe = dataframe[list_of_predictors + [outcome_variable]].copy()
    
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
    
    # Get number of unique outcomes
    number_of_unique_outcomes = len(dataframe[outcome_variable].unique())
    
    # If there are more than two outcomes, conduct multinomial regression
    if number_of_unique_outcomes > 2:
        # Create multinomial regression model
        if add_constant:
            model = sm.MNLogit(dataframe[outcome_variable], dataframe[['const'] + list_of_predictors])
        else:
            model = sm.MNLogit(dataframe[outcome_variable], dataframe[list_of_predictors])
        try:
            model_res = model.fit()
        except:
            model = sm.MNLogit(dataframe[outcome_variable], dataframe[list_of_predictors])
            model_res = model.fit()
    else:
        # Create binomial regression model
        if add_constant:
            model = sm.Logit(dataframe[outcome_variable], dataframe[['const'] + list_of_predictors])
        else:
            model = sm.Logit(dataframe[outcome_variable], dataframe[list_of_predictors])
        try:
            model_res = model.fit()
        except:
            model = sm.Logit(dataframe[outcome_variable], dataframe[list_of_predictors])
            model_res = model.fit()
    model_summary = model_res.summary()
    
   # If requested, show diagnostic plots
    if show_diagnostic_plots_for_each_predictor:
        for variable in list_of_predictors:
            fig = plt.figure(figsize=(12, 8))
            sns.regplot(
                x=variable, 
                y=outcome_variable, 
                data=dataframe,
                logistic=True, 
                y_jitter=.03
            )
            # Show plots
            plt.show()
    
    # If requested, show help text
    if show_help:
        print(
            "Quick guide on accessing output of ConductLogisticRegressionAnalysis function:",
            "\nThe ouput of the ConductLogisticRegressionAnalysis function is a dictionary containing the regression results, a test dataset of predictors, and a test dataset of outcomes.",
            "\n\t--To access the logistic regression model, use the 'Fitted Model' key."
            "\n\t--To view the model's statistical summary, use the 'Model Summary' key."
        )
    
    # Create dictionary of objects to return
    dict_return = {
        "Fitted Model": model_res,
        "Model Summary": model_summary
    }
    return dict_return

