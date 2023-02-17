import numpy as np
import sympy as sp

# Write a function that converts a matrix to row echelon form.
def ConvertMatrixToRowEchelonForm(matrix,
                                  show_pivot_columns=False):
    # If matrix is list, convert it to a matrix
    if type(matrix) == list:
        matrix = np.array(matrix)
    
    # Convert matrix to row echelon form
    matrix_row_echelon = sp.Matrix(matrix).rref()
    
    # Show rank of matrix
    print("Rank of matrix: " + str(len(matrix_row_echelon[1])))
    
    # Print list of pivot columns, if requested
    if show_pivot_columns:
        print("Pivot columns: " + str(matrix_row_echelon[1]))
    
    # Convert back to numpy array
    matrix_row_echelon = np.array(matrix_row_echelon[0])
    
    # Return matrix in row echelon form
    return matrix_row_echelon


# # Test the function
# ConvertMatrixToRowEchelonForm(
#     matrix=[
#         [4, -3, 1,  -10.],
#         [2, 1,  3,  0.],
#         [-1,    2,  -5, 17]
#     ],
# )
