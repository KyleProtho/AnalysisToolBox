# Load packages
import numpy as np

# Declare function
def ConvertSystemOfEquationsToMatrix(coefficients,
                                     constants,
                                     show_determinant=True):
    """
    Construct an augmented matrix from a system of linear equations.

    This function facilitates the transition from a standard algebraic representation 
    of a system of equations to a matrix-based representation. It combines a matrix 
     of coefficients with a vector of constants to create an augmented matrix, which 
    is the standard input for many linear algebra algorithms.

    Converting equations to matrices is essential for:
      * Formulating linear regression models and finding optimal coefficients
      * Modeling resource allocation and budget constraints for business planning
      * Evaluating the consistency of multi-variable economic or financial models
      * Preparing structural data for high-performance optimization solvers
      * Analyzing the sensitivity of output variables to changes in input parameters
      * Solving complex supply chain and production capacity equations
      * Identifying over-determined or under-determined systems in experimental data

    The function optionally calculates the determinant of the coefficient matrix. 
    A non-zero determinant indicates a non-singular system with a unique solution, 
    while a zero determinant suggests a singular system where equations may be 
    linearly dependent or inconsistent.

    Parameters
    ----------
    coefficients
        The coefficients of the variables in the system of equations. Can be a 
        nested list or a square 2D numpy.ndarray.
    constants
        The constant values (right-hand side) of the equations. Can be a list or 
        a 1D numpy.ndarray.
    show_determinant
        Whether to calculate and print the determinant and system status 
        (singular vs. non-singular). Defaults to True.

    Returns
    -------
    np.ndarray
        The augmented matrix combining coefficients and constants.

    Examples
    --------
    # Convert a system of 2 equations: 2x + 3y = 8 and 1x - 1y = 2
    coeffs = [[2, 3], [1, -1]]
    consts = [8, 2]
    augmented = ConvertSystemOfEquationsToMatrix(coeffs, consts)

    # Convert a 3x3 system and suppress the determinant output
    import numpy as np
    coeffs_3x3 = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    consts_3 = [10, 20, 30]
    matrix = ConvertSystemOfEquationsToMatrix(
        coeffs_3x3, 
        consts_3, 
        show_determinant=False
    )

    """
    
    # If the coefficients are a list of lists, convert them to a numpy array
    if type(coefficients) == list:
        coefficients = np.array(coefficients)
        
    # Calculate the determinant of the coefficient matrix
    if show_determinant:
        determinant = np.linalg.det(coefficients)
        print("Determinant:", '{:f}'.format(determinant))
        if determinant == 0:
            print("The system of equations is singular, and does not have a unique solution. At least two equations are linearly dependent.")
        else:
            print("The system of equations is non-singular, and has a unique solution. The equations are linearly independent.")
    
    # If the constants are a list, convert them to a numpy array
    if type(constants) == list:
        constants = np.array(constants)
    
    # Add the constants to the coefficients
    matrix = np.hstack((coefficients, constants.reshape(-1, 1)))
    
    # Return the matrix
    return matrix

