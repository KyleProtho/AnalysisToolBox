# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Declare function
def CreateLogisticRegressionModel(dataframe,
                                  outcome_variable,
                                  list_of_predictor_variables,
                                  scale_predictor_variables=False,
                                  # Output arguments
                                  print_peak_to_peak_range_of_each_predictor=False,
                                  test_size=0.2,
                                  show_classification_plot=True,
                                  lambda_for_regularization=0.001,
                                  max_iterations=1000,
                                  random_seed=412):
    """
    Train, evaluate, and visualize a logistic regression model for binary classification.

    This function utilizes scikit-learn's `LogisticRegression` to model the 
    probability of a discrete outcome (such as success/failure or yes/no) given 
    a set of independent predictor variables. It handles automated data cleaning, 
    optional feature scaling, and generates a confusion matrix heatmap to assess 
    classification accuracy and error patterns.

    Logistic regression is essential for:
      * Predicting the likelihood of customer conversion or subscription renewal
      * Assessing the probability of default in credit risk analysis
      * Identifying the drivers behind binary choices in consumer behavior
      * Classifying medical or technical reports into specific categories (e.g., critical/standard)
      * Forecasting the outcome of competitive bids or project approvals
      * Evaluating the impact of different features on a categorical classification
      * Building efficient baseline classifiers for machine learning pipelines

    The function provides control over regularization strength and iteration 
    limits, ensuring model convergence even with complex datasets. It 
    automatically handles complete case analysis by removing rows with missing or 
    infinite values and offers integrated visualization of the accuracy score 
    and confusion results.

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing the training and testing data.
    outcome_variable
        The name of the categorical target column to be predicted.
    list_of_predictor_variables
        A list of column names used as features for the classification.
    scale_predictor_variables
        If True, scales the features using `StandardScaler` prior to training. 
        Defaults to False.
    print_peak_to_peak_range_of_each_predictor
        If True, prints the statistical range of each predictor column to 
        monitor data dispersion. Defaults to False.
    test_size
        The proportion of the dataset used for testing. Defaults to 0.2.
    show_classification_plot
        Whether to display a heatmap of the confusion matrix labeled with 
        the overall accuracy score. Defaults to True.
    lambda_for_regularization
        The regularization parameter. Note: internally mapped to C = 1 - lambda. 
        Lower values increase regularization strength. Defaults to 0.001.
    max_iterations
        The maximum number of iterations allowed for the solver to converge. 
        Defaults to 1000.
    random_seed
        Controls the randomness of the data split and solver initialization. 
        Defaults to 412.

    Returns
    -------
    sklearn.linear_model.LogisticRegression or dict
        If `scale_predictor_variables` is False, returns the fitted 
        LogisticRegression model. If True, returns a dictionary containing 
        both the 'model' and the 'scaler' object.

    Examples
    --------
    # Create a basic logistic model to predict customer churn
    model = CreateLogisticRegressionModel(
        df, 
        outcome_variable='is_churn', 
        list_of_predictor_variables=['tenure', 'monthly_spend']
    )

    # Build a scaled model with custom regularization and iteration limits
    results = CreateLogisticRegressionModel(
        credit_df,
        outcome_variable='default_status',
        list_of_predictor_variables=['income', 'debt_ratio', 'age'],
        scale_predictor_variables=True,
        lambda_for_regularization=0.01,
        max_iterations=2000
    )

    """
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.dropna(inplace=True)
    # print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
    # Scale the predictors, if requested
    if scale_predictor_variables:
        # Scale predictors
        scaler = StandardScaler()
        dataframe[list_of_predictor_variables] = scaler.fit_transform(dataframe[list_of_predictor_variables])
        
    # Show the peak-to-peak range of each predictor
    if print_peak_to_peak_range_of_each_predictor:
        print("\nPeak-to-peak range of each predictor:")
        print(np.ptp(dataframe[list_of_predictor_variables], axis=0))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create logistic regression model
    model = LogisticRegression(
        max_iter=max_iterations, 
        random_state=random_seed,
        C=1-lambda_for_regularization,
        fit_intercept=True
    )
    
    # Train the model using the training sets and show fitting summary
    model.fit(train[list_of_predictor_variables], train[outcome_variable])
    print(f"\nNumber of iterations completed: {model.n_iter_}")
    
    # Show parameters of the model
    b_norm = model.intercept_
    w_norm = model.coef_
    print(f"\nModel parameters:    w: {w_norm}, b:{b_norm}")
    
    # Predict the test data
    test['Predicted'] = model.predict(test[list_of_predictor_variables])
    
    # Calculate the accuracy score
    score = model.score(test[list_of_predictor_variables], test[outcome_variable])

    # Print the confusion matrix
    confusion_matrix = metrics.confusion_matrix(
        test[outcome_variable], 
        test['Predicted']
    )
    if show_classification_plot:
        plt.figure(figsize=(9,9))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt=".3f", 
            linewidths=.5, 
            square=True, 
            cmap='Blues_r'
        )
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15)
        plt.show()
    else:
        print("Confusion matrix:")
        print(confusion_matrix)
        
    # Return the model
    if scale_predictor_variables:
        dict_return = {
            'model': model,
            'scaler': scaler
        }
        return(dict_return)
    else:
        return(model)

