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
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.dropna(inplace=True)
    print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
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

