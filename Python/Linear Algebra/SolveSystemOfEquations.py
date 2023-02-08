import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def SolveSystemOfEquations(coefficients,
                           constants=None,
                           show_plot=True,
                           plot_boundary=10):
    """_summary_
    This function plots a system of equations.
    
    Args:
        coefficients (np.array): A matrix of coefficients for the system of equations
        constants (np.array): A 1D array of constants for the system of equations
    """
    # If constants are not provided, set them to 0
    if constants is None:
        constants = np.zeros(coefficients.shape[0])
    
    # Calculate the determinant of the coefficient matrix
    determinant = np.linalg.det(coefficients)
    print("Determinant:", str(determinant))
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
                x = np.linspace(0, plot_boundary, 100)
                y = (constants[i] - coefficients[i, 0] * x) / coefficients[i, 1]
                
                # Generate name for the equation
                equation_name = str(coefficients[i, 0]) + "x + " + str(coefficients[i, 1]) + "y = " + str(constants[i])
                print(equation_name)
                plt.plot(x, y, label=equation_name)
        
            # Create legend
            plt.legend()

            # Show plot
            plt.show()
        
        else:
            print("\nCannot plot system of equations with more than 2 variables.")

