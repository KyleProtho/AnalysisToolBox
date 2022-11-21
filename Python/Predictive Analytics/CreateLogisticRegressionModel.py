import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def CreateLogisticRegressionModel(dataframe,
                                  outcome_variable,
                                  list_of_predictor_variables,
                                  scale_predictor_variables=True,
                                  test_size=0.2,
                                  show_classification_plot=True,
                                  lambda_for_regularization=0.001,
                                  max_iterations=1000,
                                  random_seed=412):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.dropna(inplace = True)
    dataframe = dataframe[np.isfinite(dataframe).all(1)]
    print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
    # Scale the predictors, if requested
    if scale_predictor_variables:
        # Show the mean and standard deviation of each predictor
        print("\nMean of each predictor:")
        print(dataframe[list_of_predictor_variables].mean())
        print("\nStandard deviation of each predictor:")
        print(dataframe[list_of_predictor_variables].std())
        
        # Scale predictors
        dataframe[list_of_predictor_variables] = StandardScaler().fit_transform(dataframe[list_of_predictor_variables])
        
    # Show the peak-to-peak range of each predictor
    print("\nPeak-to-peak range of each predictor:")
    print(np.ptp(dataframe[list_of_predictor_variables], axis=0))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create logistic regression model
    regr = LogisticRegression(
        max_iter=max_iterations, 
        random_state=random_seed,
        C=1-lambda_for_regularization,
        fit_intercept=True
    )
    
    # Train the model using the training sets and show fitting summary
    regr.fit(train[list_of_predictor_variables], train[outcome_variable])
    print(f"\nNumber of iterations completed: {regr.n_iter_}")
    
    # Show parameters of the model
    b_norm = regr.intercept_
    w_norm = regr.coef_
    print(f"\nModel parameters:    w: {w_norm}, b:{b_norm}")
    
    # Predict the test data
    test['Predicted'] = regr.predict(test[list_of_predictor_variables])
    
    # Print the accuracy
    score = regr.score(test[list_of_predictor_variables], test[outcome_variable])
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(score))
    
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
    return regr

# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# # iris = iris[iris['species'] != 2]
# logistic_reg_model = CreateLogisticRegressionModel(
#     dataframe=iris,
#     outcome_variable='species',
#     list_of_predictor_variables=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# )
