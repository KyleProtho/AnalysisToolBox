import numpy as np

def MultiplyRowOfMatrix(matrix, row, scalar):
    """_summary_
    The function multiplies a row of a matrix by a scalar.
    
    Args:
        matrix (np.array, list or list of lists): _description_
        row (int or float): The index of the row to be multiplied
        scalar (_type_): The scalar to multiply the row by

    Returns:
        np.array: A matrix with the row multiplied by the scalar
    """
    # If the matrix is a list of lists, convert it to a numpy array
    if type(matrix) == list:
        matrix = np.array(matrix)
    
    # Multiply the row by the scalar
    matrix[row] = matrix[row] * scalar
    
    # Return the matrix
    return matrix


# # Test the function
# MultiplyRowOfMatrix(
#     matrix=[
#         [4, -3, 1,  -10.],
#         [2, 1,  3,  0.],
#         [-1,    2,  -5, 17]
#     ],
#     row=2, 
#     scalar=2
# )