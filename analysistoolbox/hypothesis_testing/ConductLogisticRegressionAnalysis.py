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
    Conduct logistic regression analysis to model categorical outcomes and assess predictor impact.

    This function performs logistic regression to examine the relationship between one or
    more predictor variables and a categorical outcome variable. It automatically
    detects whether to perform binomial logistic regression (for two categories) or
    multinomial logistic regression (for three or more categories) using `statsmodels`.

    Logistic regression analysis is essential for:
      * Predicting binary outcomes such as customer churn, conversion, or default
      * Modeling multi-class choices like product preference or tier selection
      * Identifying key drivers that increase or decrease the probability of an event
      * Clinical research for disease diagnosis and risk factor assessment
      * Analyzing social science survey data with nominal or ordinal categories
      * Evaluating the odds ratios of specific predictors in classification tasks
      * Validating classification models through diagnostic visualizations

    The function provides statistical summaries including coefficients, p-values, and
    pseudo R-squared values. It handles data cleaning (removing NaN and infinite
    values), supports optional feature scaling, and can generate logistic regression
    plots to visualize the probability curves for each predictor.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the data for the regression analysis.
    outcome_variable
        Name of the column in the DataFrame representing the categorical outcome variable.
    list_of_predictors
        A list of column names for the independent (predictor) variables to include in
        the model.
    add_constant
        If True, adds an intercept (constant term) to the model. Defaults to True.
    scale_predictors
        If True, scales the predictor variables to have a mean of 0 and standard
        deviation of 1 using `StandardScaler` before fitting the model. Defaults to False.
    show_diagnostic_plots_for_each_predictor
        If True, displays logistic regression plots with probability curves for each
        predictor. Defaults to True.
    show_help
        If True, prints a quick guide to the console explaining how to access and
        interpret the output dictionary. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing the regression results with the following keys:
          * 'Fitted Model': The results object from the statsmodels Logit or MNLogit fit.
          * 'Model Summary': A comprehensive statistical summary of the regression model.

    Examples
    --------
    # Binomial logistic regression: Predicting customer churn
    import pandas as pd
    df = pd.DataFrame({
        'churn': [0, 1, 0, 1, 0, 1] * 20,
        'monthly_spend': [50, 100, 45, 120, 60, 150] * 20,
        'tenure_months': [24, 2, 36, 1, 12, 4] * 20
    })
    results = ConductLogisticRegressionAnalysis(
        dataframe=df,
        outcome_variable='churn',
        list_of_predictors=['monthly_spend', 'tenure_months']
    )
    print(results['Model Summary'])

    # Multinomial logistic regression: Product choice analysis
    choice_df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'A', 'B', 'C'] * 20,
        'price_sensitive': [1, 0, 0, 1, 0, 1] * 20,
        'age': [25, 45, 30, 55, 40, 35] * 20
    })
    results = ConductLogisticRegressionAnalysis(
        dataframe=choice_df,
        outcome_variable='product',
        list_of_predictors=['price_sensitive', 'age'],
        add_constant=True,
        scale_predictors=True
    )

    # Bivariate analysis with visualization
    results = ConductLogisticRegressionAnalysis(
        dataframe=df,
        outcome_variable='churn',
        list_of_predictors=['monthly_spend'],
        show_diagnostic_plots_for_each_predictor=True
    )

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

