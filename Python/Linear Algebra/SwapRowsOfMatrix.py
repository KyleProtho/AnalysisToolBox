import numpy as np

def SwapRowsOfMatrix(matrix,
                     row1,
                     row2):
    """_summary_
    This function swaps two rows of a matrix.
    
    Args:
        matrix (list or np.array): The matrix to swap the rows of.
        row1 (int): The index of the first row to swap.
        row2 (int): The index of the second row to swap.

    Returns:
        np.array(object): The matrix with the rows swapped.
    """
    # If the matrix is a list of lists, convert it to a numpy array
    if type(matrix) == list:
        matrix = np.array(matrix)
    
    # Swap the rows
    matrix[[row1, row2]] = matrix[[row2, row1]]
    
    # Return the matrix
    return matrix

# # Test the function
# SwapRowsOfMatrix(
#     matrix=[
#         [4, -3, 1,  -10.],
#         [2, 1,  3,  0.],
#         [-1,    2,  -5, 17]
#     ], 
#     row1=0, 
#     row2=2
# )
