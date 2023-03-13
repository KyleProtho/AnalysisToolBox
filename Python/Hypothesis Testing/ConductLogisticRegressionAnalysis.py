import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns

def ConductLogisticRegressionAnalysis(dataframe,
                                      outcome_variable,
                                      list_of_predictors,
                                      scale_predictors=False,
                                      show_diagnostic_plots_for_each_predictor=True,
                                      show_help=True):
    
    # Select columns specified
    dataframe = dataframe[list_of_predictors + [outcome_variable]].copy()
    
    # Remove NAN and inf values
    dataframe.dropna(inplace=True)
    dataframe = dataframe[np.isfinite(dataframe).all(1)]

    # Add constant
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
        model = sm.MNLogit(dataframe[outcome_variable], dataframe[['const'] + list_of_predictors])
        try:
            model_res = model.fit()
        except:
            model = sm.MNLogit(dataframe[outcome_variable], dataframe[list_of_predictors])
            model_res = model.fit()
    else:
        # Create binomial regression model
        model = sm.Logit(dataframe[outcome_variable], dataframe[['const'] + list_of_predictors])
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


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# # iris = iris[iris['species'] != 2]
# logistic_reg_model = ConductLogisticRegressionAnalysis(
#     dataframe=iris,
#     outcome_variable='species',
#     list_of_predictors=['petal width (cm)', 'petal length (cm)']
# )
# logistic_reg_model['Model Summary']
