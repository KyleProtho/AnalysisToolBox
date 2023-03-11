import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define a function that finds the derivative of a function
def FindDerivative(f_of_x, 
                   point,
                   step=0.0001,
                   plot_function=True,
                   plot_derivative_function=True,
                   x_minimum=-10,
                   x_maximum=10,
                   n=100,
                   tangent_line_window=None,
                   return_derivative_values=False):
    """_summary_
    This function finds the derivative of a function at a given point. It also plots the function and the tangent line at the point of interest.

    Args:
        f_of_x (lambda): The function of x.
        point (int or float): The point at which to find the derivative.
        step (float, optional): The step size to use when calculating the derivative. Defaults to 0.0001.
        plot_function (bool, optional): Whether or not to plot the function. Defaults to True.
        x_minimum (int, optional): The minimum value of x to use when plotting the function. Defaults to -10.
        x_maximum (int, optional): The maximum value of x to use when plotting the function. Defaults to 10.
        n (int, optional): The number of points to use when plotting the function. Defaults to 100.
        tangent_line_window (_type_, optional): The window to use when plotting the tangent line. Defaults to None, which will use 1/5 of the x range.
    """
    # Calculate the derivative
    try:
        derivative = (f_of_x(point + step) - f_of_x(point)) / step
    except ZeroDivisionError:
        print("The derivative at x={0} is undefined.".format(point))
        x = np.linspace(x_minimum, x_maximum, n)
        y = f_of_x(x)
        y_derivative = np.gradient(y)
        # Plot the function if requested
        if plot_function:
            plt.plot(x, y, color="black")
            plt.show()
            plt.clf()
        # Plot the derivative function if requested
        if plot_derivative_function:
            plt.plot(x, y_derivative, color="red", alpha=0.25)
            plt.title("Derivative of f(x)")
            plt.show()
        if return_derivative_values:
            return(y_derivative)
        else:
            return None
    print("The derivative at x={0} is {1}".format(point, derivative))
    
    # Plot the function 
    x = np.linspace(x_minimum, x_maximum, n)
    y = f_of_x(x)
    y_derivative = np.gradient(y)
    if plot_function:
        # Plot the function
        plt.plot(x, y, color="black")
        
        # Plot point at the point of interest
        plt.plot(point, f_of_x(point), "ro")
        
        # Create tangent line
        if tangent_line_window==None:
            tangent_line_window = (x_maximum - x_minimum) / 5
        x_tangent = np.linspace(point - tangent_line_window, point + tangent_line_window, 3)
        y_tangent = derivative * (x_tangent - point) + f_of_x(point)
        plt.plot(x_tangent, y_tangent, color="red")
        plt.show()
        plt.clf()
        
        # Plot the derivative function
        if plot_derivative_function:
            plt.plot(x, y_derivative, color="red", alpha=0.25)
            plt.title("Derivative of f(x)")
            plt.show()
        
        # Return derivative values if requested
        if return_derivative_values:
            return(y_derivative)


# # Test the function
# FindDerivative(
#     f_of_x=lambda x: x**2, 
#     point=0
# )
# # FindDerivative(
# #     f_of_x=lambda x: x**2, 
# #     point=2
# # )
# # FindDerivative(
# #     f_of_x=lambda x: x**3,
# #     point=1,
# #     x_minimum=-5,
# #     x_maximum=5
# # )
# # FindDerivative(
# #     f_of_x=lambda x: 1/x, 
# #     point=1
# # )
# # FindDerivative(
# #     f_of_x=lambda x: 1/x, 
# #     point=0
# # )
