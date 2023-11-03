import matplotlib.pyplot as plt
import numpy as np
import sympy

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
        _type_: _description_
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
        # Create y values for original function using sympy
        y_values_original = np.zeros(len(x_values))
        for i in range(len(x_values)):
            y_values_original[i] = f_of_x.evalf(subs={x: x_values[i]})
        # Create y values for derivative function
        y_values_derivative = np.zeros(len(x_values))
        for i in range(len(x_values)):
            y_values_derivative[i] = d_f_of_x.evalf(subs={x: x_values[i]})
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


# # Test the function
# # Set the variable x as a sympy symbol
# x = sympy.Symbol('x')
# FindDerivative(f_of_x=x**3 + 2*x**2 + x + 1,
#                return_derivative_function=True)
# # FindDerivative(f_of_x=x * np.e**x,
# #                return_derivative_function=True)
# # der_f_of_x = FindDerivative(
# #     f_of_x=np.e**(2*x),
# #     return_derivative_function=True
# # )
