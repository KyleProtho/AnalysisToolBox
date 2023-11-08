# Load packages
import matplotlib.pyplot as plt
import numpy as np

# Declare functions
def CalculateEigenvalues(matrix, 
                         show_plot=True, 
                         matrix_color='#033dfc', 
                         eigenvector_color='#e82a53', 
                         x_min=-1, 
                         x_max=5, 
                         y_min=-1, 
                         y_max=5, 
                         show_labels=True, 
                         plot_with_grid=True):
    """
    Calculates the eigenvalues and eigenvectors of a 2x2 matrix and optionally plots the matrix and eigenvectors.
    
    Args:
        matrix (numpy.ndarray or list): The 2x2 matrix to calculate the eigenvalues and eigenvectors of.
        show_plot (bool, optional): Whether to show a plot of the matrix and eigenvectors. Defaults to True.
        matrix_color (str, optional): The color to use for the matrix plot. Defaults to '#033dfc'.
        eigenvector_color (str, optional): The color to use for the eigenvector plot. Defaults to '#e82a53'.
        x_min (int, optional): The minimum x value for the plot. Defaults to -1.
        x_max (int, optional): The maximum x value for the plot. Defaults to 5.
        y_min (int, optional): The minimum y value for the plot. Defaults to -1.
        y_max (int, optional): The maximum y value for the plot. Defaults to 5.
        show_labels (bool, optional): Whether to show labels on the plot. Defaults to True.
        plot_with_grid (bool, optional): Whether to show a grid on the plot. Defaults to True.
    
    Returns:
        None
    
    Raises:
        ValueError: If the matrix is not 2x2.
    
    """
    
    # If matrix is list, convert to numpy array
    if type(matrix) == list:
        matrix = np.array(matrix)
    
    # Calculate eigenvalues and eigenvectors of a matrix
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # If the plot is to be shown, plot the eigenvectors and the matrix
    if show_plot:
        # Ensure that matrix is 2x2
        if matrix.shape != (2, 2):
            print('Matrix must be 2x2 to plot')
        else:
            # Perform matrix multiplication
            transformed_square = np.dot(matrix, np.array([[0,1,1,0,0],[0,0,1,1,0]]))

            # Plot the vectors
            _, ax = plt.subplots(figsize=(10, 10))
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            
            # Set the x and y limits
            ax.set_xlim([x_min-1, x_max+1])
            ax.set_ylim([y_min-1, y_max+1])

            # Add a grid, if requested
            if plot_with_grid:
                plt.grid()
            
            # Set aspect ratio to equal
            plt.gca().set_aspect("equal")
            
            # Remove borders
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            
            # Add line at x=0 and y=0
            plt.axhline(y=0, color='k', alpha=0.25)
            plt.axvline(x=0, color='k', alpha=0.25)
            
            # Plot the unit square
            plt.plot([0,1,1,0,0],[0,0,1,1,0],
                    color='black',
                    alpha=0.5)
            plt.fill([0,1,1,0,0],[0,0,1,1,0],
                    color='black',
                    alpha=0.1)
            # Add label for unit square
            if show_labels:
                plt.text(
                    0.5, 
                    -0.15, 
                    "Unit Square", 
                    fontsize=8, 
                    fontfamily='Arial', 
                    ha='center', 
                    va='center'
                )
            
            # Plot the transformed unit square
            plt.plot(
                transformed_square[0,:],
                transformed_square[1,:],
                color=matrix_color,
                alpha=0.25
            )
            plt.fill(
                transformed_square[0,:],
                transformed_square[1,:],
                color=matrix_color,
                alpha=0.1
            )
            # Add label for transformed unit square
            if show_labels:
                plt.text(
                    transformed_square[0,2]+0.15,
                    transformed_square[1,2]+0.15, 
                    "Transformed Unit Square #1", 
                    fontsize=8, 
                    fontfamily='Arial', 
                    ha='center', 
                    va='center', 
                    color=matrix_color
                )
            
            # If eigenvalues are complex, print a warning
            if ~np.iscomplex(eigenvalues).any():
                # Plot the eigenvector
                plt.quiver(
                    [0, 0],
                    [0, 0],
                    [eigenvectors[0], eigenvectors[0]], 
                    [eigenvectors[1], eigenvectors[1]], 
                    color=eigenvector_color, 
                    angles='xy', 
                    scale_units='xy', 
                    scale=1
                )
                
            # Show the plot
            plt.show()
    
    # Print the eigenvalues
    print('Eigenvalues: {}'.format(eigenvalues))
    
    # Print the eigenvectors
    print('Eigenvectors: {}'.format(eigenvectors))
    
    # If matrix is an identity matrix, print a warning
    if np.array_equal(matrix, np.identity(2)):
        print('Warning: Matrix is an identity matrix, meaning any vector is an eigenvector.')
    
    # If eigenvalues are complex, print a warning
    if np.iscomplex(eigenvalues).any():
        print('Warning: Eigenvalues are complex, meaning there are no real eigenvectors.')
    
    # If eignvectors are equivalent to identity matrix, print a warning
    if np.array_equal(eigenvectors, np.identity(len(eigenvalues))):
        print('Warning: Eigenvectors are equivalent to identity matrix, meaning any vector is an eigenvector.')
    
    # Calculate the determinant of the coefficient matrix
    determinant = np.linalg.det(eigenvectors)
    if determinant == 0:
        print("The system of eigenvector equations is singular, and does not have a unique solution. At least two equations are linearly dependent.")
    else:
        print("The system of eigenvector equations is non-singular, and has a unique solution. The equations are linearly independent.")

