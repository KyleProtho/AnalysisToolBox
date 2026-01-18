# Load packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Declare function
def SolveSystemOfEquations(coefficients,
                           constants=None,
                           show_plot=True,
                           plot_boundary=10):
    """
    Solve and visualize a system of linear algebraic equations.

    This function determines the solution to a system of equations represented by a 
    coefficient matrix and a constants vector. It evaluates the system's 
    determinant to assess singularity and, for 2D systems, provides a visual plot 
    of the lines to highlight their intersection point (the solution).

    Solving and plotting systems of equations is essential for:
      * Finding intersection points of multiple linear constraints in business models
      * Analyzing market equilibrium in supply and demand economics
      * Determining unique solutions for multi-variable experimental datasets
      * Identifying redundant or inconsistent constraints in optimization problems
      * Modeling resource allocation and balanced production planning
      * Visualizing decision boundaries for simple linear classification tasks
      * Assessing system stability through determinant analysis

    The function handles both list and NumPy array inputs. If the system is 2D, 
    it renders a plot showing the individual linear equations and their spatial 
    relationship. For higher-dimensional systems, it provides the numerical 
    solution without a plot.

    Parameters
    ----------
    coefficients
        A matrix containing the coefficients of the variables in the system. 
        Expected as a nested list or a square 2D numpy.ndarray.
    constants
        A 1D array or list representing the constants on the right-hand side of 
        the equations. If None, it defaults to a zero vector (homogeneous system).
    show_plot
        Whether to generate a 2D visualization of the equations. Only applicable 
        for systems with exactly two variables. Defaults to True.
    plot_boundary
        The range (from -value to +value) used for the x-axis in the plot. 
        Defaults to 10.

    Returns
    -------
    None
        The function prints the determinant and solution to the console and 
        displays a plot if requested and applicable.

    Examples
    --------
    # Solve a 2x2 system: 2x + 1y = 10 and 1x - 1y = 2
    coeffs = [[2, 1], [1, -1]]
    consts = [10, 2]
    SolveSystemOfEquations(coeffs, consts)

    # Solve a 3x3 system without plotting
    coeffs_3x3 = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
    consts_3x3 = [6, -4, 27]
    SolveSystemOfEquations(coeffs_3x3, consts_3x3, show_plot=False)

    """    
    # Convert coefficients to a numpy array
    coefficients = np.array(coefficients)
    
    # If constants are not provided, set them to 0
    if constants is None:
        constants = np.zeros(coefficients.shape[0])
    else:
        constants = np.array(constants)
    
    # Calculate the determinant of the coefficient matrix
    determinant = np.linalg.det(coefficients)
    print("Determinant:", '{:f}'.format(determinant))
    if determinant == 0:
        print("The system of equations is singular, and does not have a unique solution. At least two equations are linearly dependent.")
    else:
        print("The system of equations is non-singular, and has a unique solution. The equations are linearly independent.")
    
    # Solve the system of equations
    if determinant != 0:
        solution = np.linalg.solve(coefficients, constants)
        print("Solution:", str(solution))
    
    if show_plot: 
        if coefficients.shape[1] == 2:
            # Plot the system of equations
            for i in range(len(constants)):
                x = np.linspace(-1*plot_boundary, plot_boundary, 100)
                y = (constants[i] - coefficients[i, 0] * x) / coefficients[i, 1]
                
                # Generate name for the equation
                equation_name = str(coefficients[i, 0]) + "x + " + str(coefficients[i, 1]) + "y = " + str(constants[i])
                plt.plot(x, y, label=equation_name)
        
            # Create legend
            plt.legend()
            
            # Show plot
            plt.show()
        
        else:
            print("\nCannot plot system of equations with more than 2 variables.")

