# Load packages
import matplotlib.pyplot as plt
import numpy as np
import sympy

# Declare function
def FindDerivative(f_of_x,
                   print_functions=False,
                   return_derivative_function=False,
                   plot_functions=True,
                   minimum_x=-10, 
                   maximum_x=10, 
                   n=100):
    """
    This function finds the derivative of a function.

    Args:
        f_of_x (function): A function of x as a sympy expression.
        print_functions (bool, optional): Whether to print the function and its derivative. Defaults to True.
        return_derivative_function (bool, optional): Whether to return the derivative function. Defaults to False.

    Returns:
        d_f_of_x (function): The derivative of the function.
    """
    
    # Compute the derivative of the higher-order function using sympy
    d_f_of_x = sympy.diff(f_of_x, x)
    
    # Print the derivative function
    if print_functions:
        print("f(x):", f_of_x)
        print("f'(x):", d_f_of_x)
        
    # Plot the derivative function if requested
    if plot_functions:
        # Create x values
        x_values = np.linspace(minimum_x, maximum_x, n)
        
        # Vectorize the original function and its derivative
        vfunc = np.vectorize(lambda val: f_of_x.evalf(subs={x: val}))
        vfunc_derivative = np.vectorize(lambda val: d_f_of_x.evalf(subs={x: val}))

        # Create y values for original function and its derivative using the vectorized functions
        y_values_original = vfunc(x_values)
        y_values_derivative = vfunc_derivative(x_values)

        # Plot the original function
        plt.plot(x_values, y_values_original, label="f(x)")
        
        # Plot the derivative function
        plt.plot(x_values, y_values_derivative, label="f'(x)")
        
        # Add a legend
        plt.legend()
        
        # Show the plot
        plt.show()
    
    # Return the derivative function if requested
    if return_derivative_function:
        return d_f_of_x
