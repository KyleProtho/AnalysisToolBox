import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def FindMinimumSquareLoss(observed_values,
                          predicted_values,
                          show_plot=True):
    """_summary_
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
    
    # Count the number of observed values.
    n = len(observed_values)
    
    # Calculate the sum of squares.
    sum_of_squares = 0
    for i in range(n):
        sum_of_squares += (observed_values[i] - predicted_values[i])**2
    
    # Calculate the minimum square loss.
    FindMinimumSquareLoss = sum_of_squares / n
    
    # Plot the observed and predicted values on a scatter plot, if requested.
    if show_plot:
        sns.scatterplot(
            x=observed_values, 
            y=predicted_values, 
            color="blue"
        )
        sns.lineplot(
            x=observed_values, 
            y=predicted_values,
            color="blue",
            alpha=0.25
        )
        # Plot "perfect" predictions.
        sns.lineplot(
            x=observed_values, 
            y=observed_values,
            color="black",
            alpha=0.25
        )
        # Add plot labels.
        plt.xlabel("Observed Values")
        plt.ylabel("Predicted Values")
        plt.title(
            "Observed vs. Predicted Values", 
            fontsize=14,
            fontweight="bold"
        )
        # Show plot
        plt.show()
    
    # Return the minimum square loss.
    return FindMinimumSquareLoss


# Test the function.
observed_values = [1, 2, 3, 4, 5]
predicted_values = [1.5, 2.5, 3.5, 4.5, 5.5]
min_loss = FindMinimumSquareLoss(observed_values, predicted_values)
print(min_loss)

