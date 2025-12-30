# Load packages
import matplotlib.pyplot as plt
import numpy as np

# Declare function
def FindDerivative(
    f_of_x: sympy.Expr,
    print_functions: bool = False,
    return_derivative_function: bool = False,
    plot_functions: bool = False,
    **plot_kwargs
) -> Optional[sympy.Expr]:
    """
    Compute and explore the symbolic derivative of a mathematical function.

    Derivatives describe how a quantity changes with respect to another — mathematically,
    they are the instantaneous *rate of change* of a function with respect to a variable.
    Symbolic differentiation produces an exact analytic expression for this change,
    rather than a numerical approximation. SymPy is used to compute the derivative expression.

    The derivative has broad interpretation outside mathematics:
      * In intelligence analysis, the derivative of a threat likelihood function with respect 
        to time or another risk factor can reveal *acceleration* in an emerging risk trend.
      * In criminal investigation time series (e.g., incident counts over time), changes in 
        slope or inflection points may signal a shift in offender behavior or an outbreak 
        pattern that warrants deeper inquiry.
      * In healthcare analytics, derivatives quantify things like the *change in a biomarker 
        per unit time* or *how a treatment effect accelerates or decelerates over time*, 
        informing early warning signals or intervention effectiveness.

    This function:
      * Symbolically differentiates `func` with respect to the given variable (or infers the
        single symbolic variable if omitted).
      * Optionally prints the original and derivative expressions.
      * Optionally plots both functions using matplotlib for visual insight into their relationship.
      * Optionally returns the symbolic derivative expression.

    Parameters
    ----------
    f_of_x
        A SymPy expression representing the analytic function (e.g., f(x) = x**2 + 3*x).
    print_functions
        If True, prints the original and derivative expressions for inspection.
    return_derivative_function
        If True, returns the symbolic derivative expression.
    show_plot
        If True, generates a plot of the original and derivative over a default or user-supplied
        domain using `plot_kwargs`.
    **plot_kwargs
        Additional keyword arguments passed to the plotting routine.

    Returns
    -------
    The symbolic derivative expression if `return_derivative_function` is True, else None.

    Examples
    --------
    # Symbolically differentiate f(x) = x^3 + 2*x^2
    x = sympy.symbols('x')
    FindDerivative(x**3 + 2*x**2, print_functions=True, return_derivative_function=True)

    Teaching Note
    -------------
    A derivative transforms a function into a new one that reflects *how fast* the original
    changes. For example, steep slopes (large derivative values) may indicate critical
    transitions. Plotting the original and derivative together often reveals features
    like maxima, minima, or inflection points that are not obvious from raw numbers alone.

    """
    # Lazy load uncommon packages
    import sympy
    
    # Compute the derivative of the higher-order function using sympy
    try:
        d_f_of_x = sympy.diff(f_of_x, x)
    except NameError:
        raise ValueError("The function must be a sympy expression. Ensure that you have imported sympy and declared x as a sympy symbol. ( e.g., x = sympy.Symbol('x') )" )
    
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
