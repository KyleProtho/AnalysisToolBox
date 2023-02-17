import numpy as np

def AddRowsOfMatrix(matrix, row1, row2, scalar=1):
    """_summary_
    The function adds a row of a matrix to another row of the matrix.
    
    Args:
        matrix (np.array, list or list of lists): _description_
        row1 (int): The index of the row to be added
        row2 (int): The index of the row to add to
        scalar (int or float): The scalar to multiply the row by

    Returns:
        np.array: A matrix with the row multiplied by the scalar
    """
    # If the matrix is a list of lists, convert it to a numpy array
    if type(matrix) == list:
        matrix = np.array(matrix)
    
    # Add the row to the other row
    matrix[row2] = matrix[row2] + matrix[row1] * scalar
    
    # Return the matrix
    return matrix

# # Test the function
# AddRowsOfMatrix(
#     matrix=[
#         [4, -3, 1,  -10.],
#         [2, 1,  3,  0.],
#         [-1,    2,  -5, 17]
#     ],
#     row1=1,
#     row2=2, 
#     scalar=1/2
# )
