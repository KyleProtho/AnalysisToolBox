# Load packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Declare function
def FindMinimumSquareLoss(
    observed_values,
    predicted_values,
    show_plot: bool = False,
    **plot_kwargs
):
    """
    Find the parameter value that minimizes squared loss between observed data
    and a model function.

    Squared loss measures the total discrepancy between observed values and
    model-predicted values by summing the squared residuals. Minimizing this
    quantity yields the parameter value that best fits the data under a
    least-squares criterion.

    In analytic practice, least-squares minimization formalizes a common
    reasoning task: choosing the explanation, model, or parameterization that
    best accounts for observed evidence while penalizing large errors more
    heavily than small ones.

    This function:
      * Constructs a squared loss function based on the difference between
        observed `y_values` and the model's predictions.
      * Symbolically differentiates the loss with respect to the model
        parameter.
      * Solves for the parameter value that minimizes total squared error.
      * Optionally prints intermediate results and visualizes the fit.

    Parameters
    ----------
    observed_values
        Observed dependent variable values (e.g., incidents, biomarker levels,
        risk scores).
    predicted_values
        Predicted dependent variable values (e.g., incidents, biomarker levels,
        risk scores).
    show_plot
        If True, plots observed data against the fitted model.
    **plot_kwargs
        Additional keyword arguments passed to the plotting routine.

    Returns
    -------
    The parameter value that minimizes the squared loss between observed data
    and the model.

    Examples
    --------
    FindMinimumSquareLoss(
        observed_values=[1, 2, 3, 4],
        predicted_values=[2.1, 4.2, 5.9, 8.3],
        show_plot=True
    )

    Teaching Note
    -------------
    Squared loss grows rapidly as errors increase. This makes least-squares
    methods especially sensitive to large deviations, which is often desirable
    when large errors correspond to analytically meaningful failures rather
    than random noise.
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

