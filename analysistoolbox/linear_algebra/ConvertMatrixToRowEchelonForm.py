# Import packages
import numpy as np

# Declare function
def ConvertMatrixToRowEchelonForm(matrix,
                                  show_pivot_columns=False):
    """
    Transform a matrix into its reduced row echelon form (RREF).

    This function utilizes the symbolic computation capabilities of SymPy to reliably 
    convert a matrix (passed as a list or NumPy array) into its reduced row echelon 
    form. The transformation is performed using Gaussian elimination with back-substitution, 
    ensuring that the leading coefficient of every non-zero row is 1, and is the only 
    non-zero entry in its column.

    Transforming a matrix to row echelon form is essential for:
      * Detecting multicollinearity and identifying redundant features in a dataset
      * Determining the rank of a data matrix to assess feature dimensionality
      * Selecting linearly independent variables for stable predictive modeling
      * Solving systems of linear constraints in business optimization problems
      * Finding a basis for the row or column space to understand data variance
      * Identifying pivot columns to pinpoint the most informative, non-redundant features
      * Analyzing the structural reachability within network adjacency matrices

    The function automatically calculates and prints the rank of the matrix. It also 
    offers the option to output the indices of the pivot columns, which correspond to 
    the linearly independent columns of the original matrix.

    Parameters
    ----------
    matrix
        The input matrix to be converted. Can be a nested list or a 2D 
        numpy.ndarray of any dimensions.
    show_pivot_columns
        If True, the function will print a list containing the indices of the 
        pivot columns (the columns containing leading 1s). Defaults to False.

    Returns
    -------
    np.ndarray
        A NumPy array representing the matrix in reduced row echelon form.

    Examples
    --------
    # Convert a 2x3 matrix to reduced row echelon form
    matrix = [[1, 2, 3], [4, 5, 6]]
    rref_matrix = ConvertMatrixToRowEchelonForm(matrix)

    # Convert a 3x3 matrix and display pivot column indices
    import numpy as np
    a = np.array([[1, 2, -1], [2, 4, -2], [3, 6, -3]])
    rref_a = ConvertMatrixToRowEchelonForm(a, show_pivot_columns=True)

    """
    # Lazy load uncommon packages
    import sympy as sp
    
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

