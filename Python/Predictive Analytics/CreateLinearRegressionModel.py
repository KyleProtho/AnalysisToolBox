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
                                list_of_predictor_variables,
                                scale_predictor_variables=True,
                                test_size=0.2,
                                max_iterations=1000,
                                learning_rate=0.01,
                                lambda_for_regularization=0.001,
                                show_plot_of_residuals=True,
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
    
    # Create linear regression object
    regr = linear_model.SGDRegressor(
        loss='squared_error',
        max_iter=max_iterations,
        eta0=learning_rate,
        alpha=lambda_for_regularization,
        random_state=random_seed,
        fit_intercept=True,
    )
    
    # Train the model using the training sets and show fitting summary
    regr.fit(train[list_of_predictor_variables], train[outcome_variable])
    print(f"\nNumber of iterations completed: {regr.n_iter_}")
    print(f"Number of weight updates: {regr.t_}")
    
    # Show parameters of the model
    b_norm = regr.intercept_
    w_norm = regr.coef_
    print(f"\nModel parameters:    w: {w_norm}, b:{b_norm}")
        
    # Show the explained variance score: 
    print("\nVariance score: %.2f" % regr.score(train[list_of_predictor_variables], train[outcome_variable]))
    print("Note: 1 is perfect prediction and 0 means that there is no linear relationship between X and Y.")
    
    # The mean squared error
    print("\nMean squared error: %.2f" % np.mean((regr.predict(test[list_of_predictor_variables]) - test[outcome_variable]) ** 2))
    
    # Plot predicted and observed outputs if requested
    if show_plot_of_residuals:
        fig, ax = plt.subplots(
            int(np.ceil(len(list_of_predictor_variables))), 
            1,
            figsize=(4, int(np.ceil(len(list_of_predictor_variables)))*4.5),
            sharey=True
        )
        for i in range(len(ax)):
            ax[i].scatter(
                train[list_of_predictor_variables[i]],
                train[outcome_variable], 
                label="Observed"
            )
            ax[i].set_xlabel(list_of_predictor_variables[i])
            ax[i].set_ylabel(outcome_variable)
            ax[i].scatter(
                train[list_of_predictor_variables[i]],
                regr.predict(train[list_of_predictor_variables]),
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
#                                                list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#                                                scale_predictor_variables=True)

