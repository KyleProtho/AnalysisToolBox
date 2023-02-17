import numpy as np

def ConvertSystemOfEquationsToMatrix(coefficents,
                                     constants):
    """_summary_
    This function converts a system of equations to a matrix.
    
    Args:
        coefficents (list or np.array): The coefficents of the system of equations. 
        constants (list or np.array): The constants of the system of equations.

    Returns:
        np.array: The system of equations as a matrix.
    """
    
    # If the coefficents are a list of lists, convert them to a numpy array
    if type(coefficents) == list:
        coefficents = np.array(coefficents)
    
    # If the constants are a list, convert them to a numpy array
    if type(constants) == list:
        constants = np.array(constants)
    
    # Add the constants to the coefficents
    matrix = np.hstack((coefficents, constants.reshape(-1, 1)))
    
    # Return the matrix
    return matrix

# # Test the function
# ConvertSystemOfEquationsToMatrix(
#     coefficents=[
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
