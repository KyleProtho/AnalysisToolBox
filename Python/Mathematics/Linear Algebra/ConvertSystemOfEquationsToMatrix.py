import numpy as np

def ConvertSystemOfEquationsToMatrix(coefficients,
                                     constants,
                                     show_determinant=True):
    """_summary_
    This function converts a system of equations to a matrix.
    
    Args:
        coefficients (list or np.array): The coefficients of the system of equations. 
        constants (list or np.array): The constants of the system of equations.

    Returns:
        np.array: The system of equations as a matrix.
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


# # Test the function
# ConvertSystemOfEquationsToMatrix(
#     coefficients=[
#         [4, -3, 1],
#         [2, 1,  3],
#         [-1,    2,  -5]
#     ],
#     constants=[
#         -10, 
#         0, 
#         17
#     ]
# )
