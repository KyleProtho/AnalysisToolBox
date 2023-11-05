# Load packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Declare function
def FindMinimumSquareLoss(observed_values,
                          predicted_values,
                          show_plot=True):
    """
    This function calculates the minimum square loss between observed and predicted values.

    Args:
    observed_values: A list of observed values.
    predicted_values: A list of predicted values.

    Returns:
    The minimum square loss between the observed and predicted values.
    """
    # Check if the number of observed values is equal to the number of predicted values.
    if len(observed_values) != len(predicted_values):
        raise ValueError("The number of observed values must be equal to the number of predicted values.")
    
    # If values are passed as a list, convert them to a numpy array.
    if type(observed_values) == list:
        observed_values = np.array(observed_values)
    if type(predicted_values) == list:
        predicted_values = np.array(predicted_values)    
    
    # Count the number of observed values.
    n = len(observed_values)
    
    # Calculate the prediction error.
    sq_prediction_error = (predicted_values - observed_values) ** 2
    
    # Calculate the sum of squares.
    sum_of_squares = sum(sq_prediction_error)
    
    # Calculate the minimum square loss.
    FindMinimumSquareLoss = sum_of_squares / n
    
    # Plot the prediction errors in a plot, if requested.
    if show_plot:
        sns.boxplot(y=sq_prediction_error, color="grey")
        # Add the minimum square loss as a horizontal line.
        plt.axhline(FindMinimumSquareLoss, color="r", linestyle="--")
        # Add labels.
        plt.title("Squared Prediction Errors", fontsize=14, fontweight="bold")
        plt.text(0.5, FindMinimumSquareLoss + .01, "Min. Sq. Loss = " + str(round(FindMinimumSquareLoss, 2)))
        # Remove borders from plot
        sns.despine(bottom=True)
        # Show plot
        plt.show()
    
    # Return the minimum square loss.
    return FindMinimumSquareLoss

