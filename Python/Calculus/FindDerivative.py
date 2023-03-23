import sympy
# Set the variable x as a sympy symbol
x = sympy.Symbol('x')

def FindDerivative(f_of_x,
                   print_functions=False,
                   return_derivative_function=False):
    """_summary_
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
    
    # Return the derivative function if requested
    if return_derivative_function:
        return d_f_of_x


# # Test the function
# FindDerivative(f_of_x=x**3 + 2*x**2 + x + 1,
#                return_derivative_function=True)

# # import numpy as np
# # FindDerivative(f_of_x=x * np.e**x,
# #                return_derivative_function=True)

# # import numpy as np
# # der_f_of_x = FindDerivative(
# #     f_of_x=np.e**(2*x),
# #     return_derivative_function=True
# # )
