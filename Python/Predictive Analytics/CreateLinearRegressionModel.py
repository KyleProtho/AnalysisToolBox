# Load pacakges
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Function to create linear regression model
def CreateLinearRegressionModel(dataframe,
                          outcome_variable,
                          list_of_predictors,
                          test_size=0.2,
                          random_seed = 412):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictors + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.dropna(inplace = True)
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(train[list_of_predictors], train[outcome_variable])
    
    # Show the coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)
    
    # Show the explained variance score: 
    print('Variance score: %.2f' % regr.score(train[list_of_predictors], train[outcome_variable]))
    print("Note: 1 is perfect prediction and 0 means that there is no linear relationship between X and Y.")
    
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((regr.predict(test[list_of_predictors]) - test[outcome_variable]) ** 2))
    
    # Plot outputs
    plt.scatter(
        test[outcome_variable], 
        regr.predict(test[list_of_predictors]),
        color='blue',
        linewidth=3
    )
    plt.xlabel('Observed ' + outcome_variable)
    plt.ylabel('Predicted ' + outcome_variable)
    plt.show()
    
    # Return the model
    return regr
