# Load pacakges
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to create linear regression model
def CreateLinearRegressionModel(dataframe,
                                outcome_variable,
                                list_of_predictors,
                                scale_predictors=False,
                                test_size=0.2,
                                max_iterations=1000,
                                learning_rate=0.0001,
                                show_plot_of_residuals=True,
                                random_seed=412):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictors + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.dropna(inplace = True)
    
    # Scale the predictors, if requested
    if scale_predictors:
        # Show the mean and standard deviation of each predictor
        print("Mean of each predictor:")
        print(dataframe[list_of_predictors].mean())
        print("\nStandard deviation of each predictor:")
        print(dataframe[list_of_predictors].std())
        
        # Scale predictors
        dataframe[list_of_predictors] = StandardScaler().fit_transform(dataframe[list_of_predictors])
        
    # Show the peak-to-peak range of each predictor
    print("\nPeak-to-peak range of each predictor:")
    print(np.ptp(dataframe[list_of_predictors], axis=0))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create linear regression object
    regr = linear_model.SGDRegressor(
        loss='squared_loss',
        max_iter=max_iterations,
        alpha=learning_rate,
        random_state=random_seed,
        fit_intercept=True,
    )
    
    # Train the model using the training sets
    regr.fit(train[list_of_predictors], train[outcome_variable])
    
    # Show parameters of the model
    b_norm = regr.intercept_
    w_norm = regr.coef_
    print(f"\nModel parameters:                   w: {w_norm}, b:{b_norm}")
        
    # Show the explained variance score: 
    print("\nVariance score: %.2f" % regr.score(train[list_of_predictors], train[outcome_variable]))
    print("Note: 1 is perfect prediction and 0 means that there is no linear relationship between X and Y.")
    
    # The mean squared error
    print("\nMean squared error: %.2f" % np.mean((regr.predict(test[list_of_predictors]) - test[outcome_variable]) ** 2))
    
    # Plot predicted and observed outputs if requested
    if show_plot_of_residuals:
        fig, ax = plt.subplots(
            int(np.ceil(len(list_of_predictors))), 
            1,
            figsize=(4, int(np.ceil(len(list_of_predictors)))*4.5),
            sharey=True
        )
        for i in range(len(ax)):
            ax[i].scatter(
                train[list_of_predictors[i]],
                train[outcome_variable], 
                label="Observed"
            )
            ax[i].set_xlabel(list_of_predictors[i])
            ax[i].set_ylabel(outcome_variable)
            ax[i].scatter(
                train[list_of_predictors[i]],
                regr.predict(train[list_of_predictors]),
                label='Predicted'
            ) 
        ax[0].legend()
        fig.suptitle("Observed and predicted outcome")
        plt.show()
    
    # Return the model
    return regr


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# linear_reg_model = CreateLinearRegressionModel(dataframe=iris,
#                                                outcome_variable='sepal length (cm)',
#                                                list_of_predictors=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#                                                scale_predictors=True)

