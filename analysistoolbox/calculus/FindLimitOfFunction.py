# Load packages
import matplotlib.pyplot as plt
import numpy as np

# Declare function
def FindLimitOfFunction(f_of_x,
                        point,
                        step=0.0001,
                        plot_function=True,
                        plot_tangent_line=True,
                        x_minimum=-10,
                        x_maximum=10,
                        n=100,
                        tangent_line_window=None):
    """
    This function finds the x_tangent of a function at a given point. 
    It also plots the function and the tangent line at the point of interest.

    Args:
    f_of_x (lambda): The function of x.
    point (int or float): The point at which to find the x_tangent.
    step (float, optional): The step size to use when calculating the x_tangent. Defaults to 0.0001.
    plot_function (bool, optional): Whether or not to plot the function. Defaults to True.
    x_minimum (int, optional): The minimum value of x to use when plotting the function. Defaults to -10.
    x_maximum (int, optional): The maximum value of x to use when plotting the function. Defaults to 10.
    n (int, optional): The number of points to use when plotting the function. Defaults to 100.
    tangent_line_window (_type_, optional): The window to use when plotting the tangent line. Defaults to None, which will use 1/5 of the x range.
    
    Returns:
    float: The x_tangent at the point of interest.
    """
    
    # Create array of values based on the function
    x = np.linspace(x_minimum, x_maximum, n)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = f_of_x(x[i])
    
    # Calculate the x_tangent
    x_tangent = (f_of_x(point+step) - f_of_x(point-step)) / (2*step)
    
    # Calculate the limit at the point of interest
    try:
        limit = f_of_x(point)
        print("The limit at x={0} is ~{1}".format(point, limit))
        # Create the tangent line
        y_tangent = f_of_x(point) + x_tangent*(x - point)
        
        # Plot the function if requested
        if plot_function:
            # Plot point at the point of interest
            plt.plot(point, f_of_x(point), "ro")
            
        # Plot the tangent line
        if plot_tangent_line:
            # plt.scatter(x0, my_function(x0), color='red')
            plt.plot(x, y_tangent, label='Tangent line', color='red', alpha=0.5)
            plt.scatter(point, f_of_x(point), color='red')
        
        # Plot the function if requested
        if plot_function:
            # Plot the function
            plt.plot(x, y, color="black", label='Function')
            
            # Add title
            plt.title("f(x)")
            
            # Add legend
            plt.legend()
            
            # Show plot
            plt.show()
            plt.clf()
        
        # Return the limit
        return limit
        
    except ZeroDivisionError:
        print("The limit at x={0} is undefined.".format(point))
