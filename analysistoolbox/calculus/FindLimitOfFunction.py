# Load packages
import matplotlib.pyplot as plt
import numpy as np
import sympy

# Declare function
def FindLimitOfFunction(
    f_of_x,
    point,
    step: float = 0.001,
    plot_function: bool = False,
    plot_tangent_line: bool = False,
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
        A symbolic (SymPy) expression or a numeric callable (e.g., lambda) representing f(x).
    point
        The input value that x approaches (e.g., 0, infinity, or a finite critical point).
    step
        Step size used for numerical approximation or plotting.
    plot_function
        If True, generate a matplotlib plot of `f_of_x` near `point`.
    plot_tangent_line
        If True, plot a tangent line at the point.
    x_minimum, x_maximum
        The range over which to plot `f_of_x`. If None, a default window around `point` is chosen.
    n
        Number of points used for plotting.
    tangent_line_window
        Width of the window around `point` over which the tangent line is drawn.
    **plot_kwargs
        Additional keyword arguments passed to the plotting routine.

    Returns
    -------
    A symbolic SymPy expression representing the limit of `f_of_x` as x→`point`.
    """

    # Identify if f_of_x is symbolic or a callable
    if isinstance(f_of_x, sympy.Expr):
        symbols = f_of_x.free_symbols
        if symbols:
            x_var = sorted(list(symbols), key=lambda s: s.name)[0]
        else:
            x_var = sympy.Symbol('x')
        
        # Calculate symbolic limit
        limit_val = sympy.limit(f_of_x, x_var, point)
        
        # Create a numerical function for plotting
        f_numeric = lambda val: float(f_of_x.evalf(subs={x_var: val}))
    else:
        # Assume it's a callable
        f_numeric = f_of_x
        # For callable, we can only approximate the limit or use the value at point
        try:
            limit_val = sympy.sympify(f_numeric(point))
        except Exception:
            # If undefined at point, try to approximate
            limit_val = sympy.sympify((f_numeric(point - step) + f_numeric(point + step)) / 2)

    # Set default plotting range if not provided
    if x_minimum is None:
        x_minimum = point - 5
    if x_maximum is None:
        x_maximum = point + 5

    # Create array of values based on the function
    x_vals = np.linspace(x_minimum, x_maximum, n)
    y_vals = np.zeros(len(x_vals))
    for i in range(len(x_vals)):
        try:
            y_vals[i] = f_numeric(x_vals[i])
        except Exception:
            y_vals[i] = np.nan
    
    # Calculate the numerical derivative (slope) for the tangent line
    try:
        slope = (f_numeric(point + step) - f_numeric(point - step)) / (2 * step)
        y_at_point = f_numeric(point)
    except Exception:
        # If point is undefined, use approximation near point
        slope = (f_numeric(point + step) - f_numeric(point - step)) / (2 * step)
        y_at_point = (f_numeric(point + step) + f_numeric(point - step)) / 2

    # Calculate the limit at the point of interest
    print(f"The limit at x={point} is ~{limit_val}")
    
    # Create the tangent line
    y_tangent = y_at_point + slope * (x_vals - point)
        
    # Plotting
    if plot_function:
        fig, ax = plt.subplots()
        
        # Plot the function
        ax.plot(x_vals, y_vals, color="black", label='Function')
        
        # Plot point at the point of interest
        try:
            ax.plot(point, f_numeric(point), "ro")
        except Exception:
            ax.plot(point, float(limit_val), "ro", fillstyle='none') # Open circle for discontinuity
            
        # Plot the tangent line if requested
        if plot_tangent_line:
            # Limit the tangent line to a window around point
            mask = (x_vals >= point - tangent_line_window) & (x_vals <= point + tangent_line_window)
            ax.plot(x_vals[mask], y_tangent[mask], label='Tangent line', color='red', alpha=0.5)
        
        ax.set_title(f"Limit of f(x) as x approaches {point}")
        ax.legend()
        plt.show()
    
    return limit_val
