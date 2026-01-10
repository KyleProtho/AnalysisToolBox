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
    Conduct linear regression analysis to model relationships between variables and assess significance.

    This function performs Ordinary Least Squares (OLS) linear regression to examine how
    one or more predictor variables influence a continuous outcome variable. It provides
    statistical summaries, regression coefficients, and diagnostic visualizations to
    help interpret the strength and nature of the relationships in the data.

    Linear regression analysis is essential for:
      * Estimating the impact of independent variables on a continuous outcome
      * Predictive modeling for numerical values and trend forecasting
      * Testing scientific and economic hypotheses regarding variable associations
      * Analyzing the drivers of business metrics like revenue or customer satisfaction
      * Social science research on the effects of demographic or environmental factors
      * Identifying key features and their relative importance in a dataset
      * Validating the assumptions of linear relationships between variables

    The function automatically handles preprocessing by removing missing (NaN) and
    infinite values. It supports optional feature scaling using `StandardScaler`
    and can generate detailed diagnostic plots for each predictor variable using
    `statsmodels` diagnostics.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the variables for the regression analysis.
    outcome_variable
        Name of the column in the DataFrame representing the dependent (outcome) variable.
    list_of_predictors
        A list of column names for the independent (predictor) variables to include in
        the model.
    add_constant
        If True, adds an intercept (constant term) to the model. Defaults to False.
    scale_predictors
        If True, scales the predictor variables to have a mean of 0 and standard
        deviation of 1 using `StandardScaler` before fitting the model. Defaults to False.
    show_diagnostic_plots_for_each_predictor
        If True, displays diagnostic regression plots (e.g., partial regression plots)
        for each predictor in the model. Defaults to False.
    show_help
        If True, prints a quick guide to the console explaining how to access and
        interpret the output dictionary. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing the regression results with the following keys:
          * 'Fitted Model': The results object from the statsmodels OLS fit.
          * 'Model Summary': A comprehensive statistical summary of the regression model.

    Examples
    --------
    # Analyze the impact of experience and education on salary
    import pandas as pd
    df = pd.DataFrame({
        'salary': [50000, 60000, 55000, 80000, 75000] * 10,
        'experience_years': [1, 5, 3, 10, 8] * 10,
        'education_level': [12, 16, 14, 18, 16] * 10
    })
    results = ConductLinearRegressionAnalysis(
        dataframe=df,
        outcome_variable='salary',
        list_of_predictors=['experience_years', 'education_level'],
        add_constant=True
    )
    print(results['Model Summary'])

    # Perform regression with feature scaling and diagnostic plots
    advertising_df = pd.DataFrame({
        'sales': [22, 10, 9, 18, 12] * 10,
        'tv_spend': [230, 44, 17, 151, 180] * 10,
        'radio_spend': [37, 39, 45, 41, 10] * 10
    })
    results = ConductLinearRegressionAnalysis(
        dataframe=advertising_df,
        outcome_variable='sales',
        list_of_predictors=['tv_spend', 'radio_spend'],
        add_constant=True,
        scale_predictors=True,
        show_diagnostic_plots_for_each_predictor=True
    )

    # Simple bivariate regression without intercept
    results = ConductLinearRegressionAnalysis(
        dataframe=df,
        outcome_variable='salary',
        list_of_predictors=['experience_years'],
        add_constant=False
    )

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

