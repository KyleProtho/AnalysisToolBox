# Import packages
import numpy as np
import sympy as sp

# Declare function
def ConvertMatrixToRowEchelonForm(matrix,
                                  show_pivot_columns=False):
    """
    Converts a matrix to row echelon form.
    
    Args:
        matrix (np.array or list): Matrix to be converted to row echelon form.
        show_pivot_columns (bool, optional): If True, the function will print the pivot columns. Defaults to False.

    Returns:
        np.array: Matrix in row echelon form.
    """
    
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

