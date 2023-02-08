import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def SolveSystemOfEquations(coefficients,
                           constants,
                           show_plot=True):
    """_summary_
    This function plots a system of equations.
    
    Args:
        coefficients (np.array): A 2D array of coefficients for the system of equations
        constants (np.array): A 1D array of constants for the system of equations
    """
    # Calculate the determinant of the coefficient matrix
    determinant = np.linalg.det(coefficients)
    print("Determinant:", str(determinant))
    if determinant == 0:
        print("The system of equations is singular, and has no solution.")
    else:
        print("The system of equations is non-singular, and has a unique solution.")
    
    # Solve the system of equations
    solution = np.linalg.solve(coefficients, constants)
    print("Solution:", str(solution))
    
    if show_plot:
        # Plot the system of equations
        for i in range(len(constants)):
            x = np.linspace(0, 10, 100)
            y = (constants[i] - coefficients[i, 0] * x) / coefficients[i, 1]
            
            # Generate name for the equation
            equation_name = str(coefficients[i, 0]) + "x + " + str(coefficients[i, 1]) + "y = " + str(constants[i])
            print(equation_name)
            plt.plot(x, y, label=equation_name)
        
        # Create legend
        plt.legend()

        # Show plot
        plt.show()
    
