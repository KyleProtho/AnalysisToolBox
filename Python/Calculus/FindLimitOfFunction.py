import matplotlib.pyplot as plt
import numpy as np

# Define a function that finds the derivative of a function
def FindLimitOfFunction(f_of_x, 
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
    # Create array of values based on the function
    x = np.linspace(x_minimum, x_maximum, n)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = f_of_x(x[i])
    
    # Calculate the derivative
    # y_derivative = np.gradient(y)
    derivative = (f_of_x(point+step) - f_of_x(point-step)) / (2*step)
    
    # Calculate the limit at the point of interest
    try:
        limit = f_of_x(point)
        print("The limit at x={0} is ~{1}".format(point, limit))
        # Create the tangent line
        y_derivative = f_of_x(point) + derivative*(x - point)
        
        # Plot the function if requested
        if plot_function:
            # Plot point at the point of interest
            plt.plot(point, f_of_x(point), "ro")
            
        # Plot the tangent line
        if plot_derivative_function:
            # plt.scatter(x0, my_function(x0), color='red')
            plt.plot(x, y_derivative, label='Tangent line', color='red', alpha=0.5)
            plt.scatter(point, f_of_x(point), color='red')
        
    except ZeroDivisionError:
        print("The limit at x={0} is undefined.".format(point))
    
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
    
    # Return derivative values if requested
    if return_derivative_values:
        return(y_derivative)


# # Test the function
# FindLimitOfFunction(
#     f_of_x=lambda x: np.power(x, 2),
#     point=0
# )
# FindLimitOfFunction(
#     f_of_x=lambda x: 1/x, 
#     point=0
# )
# FindLimitOfFunction(
#     f_of_x=lambda x: np.sin(x), 
#     point=0
# )
# FindLimitOfFunction(
#     f_of_x=lambda x: np.exp(1) ** x, 
#     point=1,
#     x_minimum=0,
#     x_maximum=10
# )
# FindLimitOfFunction(
#     f_of_x=lambda x: np.log(x), 
#     point=1,
#     x_minimum=0,
#     x_maximum=10
# )

