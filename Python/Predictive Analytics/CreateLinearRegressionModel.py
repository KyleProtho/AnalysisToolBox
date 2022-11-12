# Load pacakges
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Function to create linear regression model
def CreateLinearRegressionModel(dataframe,
                                outcome_variable,
                                list_of_predictors,
                                scale_predictors=False,
                                test_size=0.2,
                                random_seed=412):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictors + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.dropna(inplace = True)
    
    # Add constant to predictors
    dataframe = sm.add_constant(dataframe)
    
    # Scale the predictors, if requested
    if scale_predictors:
        # Show the mean and standard deviation of each predictor
        print("Mean of each predictor:")
        print(dataframe[list_of_predictors].mean())
        print("\nStandard deviation of each predictor:")
        print(dataframe[list_of_predictors].std())
        
        # Scale predictors
        dataframe[list_of_predictors] = dataframe[list_of_predictors].apply(lambda x: (x - x.mean()) / x.std())
    
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
    
    # Show summary of linear regression model
    model_summary = sm.OLS(train[outcome_variable], train[['const'] + list_of_predictors]).fit()
    model_summary = model_summary.summary()
    print(model_summary)
    
    # Show the explained variance score: 
    print("Variance score: %.2f" % regr.score(train[list_of_predictors], train[outcome_variable]))
    print("Note: 1 is perfect prediction and 0 means that there is no linear relationship between X and Y.")
    
    # The mean squared error
    print("\nMean squared error: %.2f" % np.mean((regr.predict(test[list_of_predictors]) - test[outcome_variable]) ** 2))
    
    # Plot predicted and observed outputs
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

# Test the function
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
linear_reg_model = CreateLinearRegressionModel(dataframe=iris,
                                               outcome_variable='sepal length (cm)',
                                               list_of_predictors=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                                               scale_predictors=True)
