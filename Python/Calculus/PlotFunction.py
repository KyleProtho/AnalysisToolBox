import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

# Define a function that plots a function
def PlotFunction(f_of_x, 
                 minimum_x=-10, 
                 maximum_x=10, 
                 n=100):
    """_summary_
    This function plots a function of x.

    Args:
        f_of_x (lambda): The function of x.
        minimum_x (int, optional): The minimum value of x to use when plotting the function. Defaults to -10.
        maximum_x (int, optional): The maximum value of x to use when plotting the function. Defaults to 10.
        n (int, optional): The number of points to use when plotting the function. Defaults to 100.
    """
    
    # Plot the function
    x = np.linspace(minimum_x, maximum_x, n)
    try:
        y = f_of_x(x)
    except TypeError:
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = f_of_x(x[i])
    plt.plot(x, y)
    
    # Show the plot
    plt.show()

# # Test the function
# PlotFunction(
#     f_of_x=lambda x: x**2
# )
# # PlotFunction(
# #     f_of_x=lambda x: math.sin(x)
# # )
