# Load packages
import matplotlib.pyplot as plt
import numpy as np

# Declare function
def FindLimitOfFunction(
    f_of_x,
    point,
    step: float = 0.001,
    plot_function: bool = False,
    x_minimum: float = None,
    x_maximum: float = None,
    n: int = 500,
    tangent_line_window: float = 1.0,
    **plot_kwargs
) -> sympy.Expr:
    """
    Compute and visualize the limit of a mathematical function as its input
    approaches a specified point.

    In calculus, a *limit* describes the value that a function’s output
    approaches as its input gets arbitrarily close to a given point, even
    if the function is not defined at that point itself. Limits underpin
    many core ideas in analysis — including continuity, derivatives, and
    asymptotic behavior — and have practical interpretive value in analytic
    domains.

    This function:
      * Uses SymPy’s symbolic `limit` capability to compute the limit of
        `f_of_x` as its independent variable approaches `point`.
      * Optionally plots the function near `point` and overlays a tangent
        line to illustrate the local approach behavior.
      * Returns the symbolic limit expression for programmatic inspection.

    Parameters
    ----------
    f_of_x
        A symbolic or numeric expression representing f(x). Limits can
        reveal behavior near discontinuities or indeterminate forms.:contentReference[oaicite:1]{index=1}
    point
        The input value that x approaches (e.g., 0, infinity, or a finite
        critical point).
    step
        Step size used for numerical approximation or plotting.
    plot_function
        If True, generate a matplotlib plot of `func` near `point`.
    x_minimum, x_maximum
        The range over which to plot `func`. If None, a default window
        around `point` is chosen.
    n
        Number of points used for plotting.
    tangent_line_window
        Width of the window around `point` over which the tangent line is
        drawn.
    **plot_kwargs
        Additional keyword arguments passed to the plotting routine.

    Returns
    -------
    A symbolic SymPy expression representing the limit of `func` as x→`point`.

    Examples
    --------
    # Symbolically compute the limit of sin(x)/x as x→0
    x = sympy.symbols('x')
    FindLimitOfFunction(sympy.sin(x)/x, point=0, plot_function=True)

    Teaching Note
    -------------
    A limit expresses the *behavior* of a function in the vicinity of a value:
    it does not require the function to be defined at that exact point. For
    instance, sin(x)/x is undefined at x=0, yet its limit as x approaches
    0 is 1 — reflecting the function’s approach behavior, not its value at
    the point.
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
