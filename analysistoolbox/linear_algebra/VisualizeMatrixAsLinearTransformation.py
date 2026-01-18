# Load packages
import matplotlib.pyplot as plt
import numpy as np

# Declare function
def VisualizeMatrixAsLinearTransformation(two_by_two_matrix1,
                                          two_by_two_matrix2=None,
                                          plot_with_grid=False,
                                          show_labels=True,
                                          matrix1_color='#033dfc',
                                          matrix2_color='#03b1fc',
                                          x_min=-5,
                                          x_max=5,
                                          y_min=-5,
                                          y_max=5):
    """
    Visualize one or two 2x2 linear transformations using a unit square.

    This function provides a geometric representation of how 2x2 matrices transform 
    the 2D plane. By plotting a "Unit Square" and its image after being multiplied 
    by the matrix, users can clearly see the effects of scaling, rotation, shearing, 
    and reflection in real-time.

    Visualizing linear transformations is essential for:
      * Understanding how matrices warp, rotate, or scale coordinate spaces
      * Identifying determinants geometrically through area changes of the square
      * Analyzing the impact of composite transformations on data orientation
      * Teaching fundamental concepts of basis changes and linear mappings
      * Evaluating structural stability by examining transformation distortions
      * Providing intuitive insights into PCA and other dimensionality reduction methods

    The function supports plotting a single transformation or a sequence of two 
    transformations. It handles list and NumPy array inputs and allows for 
    extensive customization of colors and plot boundaries.

    Parameters
    ----------
    two_by_two_matrix1
        The first 2x2 matrix to visualize. Expected as a nested list or a 2D 
        numpy.ndarray.
    two_by_two_matrix2
        An optional second 2x2 matrix. If provided, the function visualizes the 
        composition of matrix2 applied to the result of matrix1. Defaults to None.
    plot_with_grid
        Whether to show a coordinate grid on the plot. Defaults to False.
    show_labels
        Whether to show descriptive labels for the unit squares. Defaults to True.
    matrix1_color
        The color for the first transformed unit square. Defaults to '#033dfc'.
    matrix2_color
        The color for the second transformed unit square (composed). Defaults 
        to '#03b1fc'.
    x_min, x_max, y_min, y_max
        The boundaries for the plot axes. Default to -5 and 5 respectively.

    Returns
    -------
    None
        The function renders an interactive Matplotlib plot.

    Examples
    --------
    # Visualize a simple 45-degree rotation
    import numpy as np
    theta = np.radians(45)
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    VisualizeMatrixAsLinearTransformation(rotation)

    # Visualize a shear transformation followed by a scaling
    shear = [[1, 2], [0, 1]]
    scale = [[2, 0], [0, 2]]
    VisualizeMatrixAsLinearTransformation(shear, scale, plot_with_grid=True)

    """
    
    # If matrix is list, convert to numpy array
    if type(two_by_two_matrix1) == list:
        two_by_two_matrix1 = np.array(two_by_two_matrix1)
    if two_by_two_matrix2 is not None:
        if type(two_by_two_matrix2) == list:
            two_by_two_matrix2 = np.array(two_by_two_matrix2)
            
    # Perform matrix multiplication
    transformed_square = np.dot(two_by_two_matrix1, np.array([[0,1,1,0,0],[0,0,1,1,0]]))
    if two_by_two_matrix2 is not None:
        transformed_square_2 = np.dot(two_by_two_matrix2, transformed_square)
        
    # Ensure matrix is 2x2
    if two_by_two_matrix1.shape != (2,2):
        raise Exception("Matrix #1 must be 2x2")
    if two_by_two_matrix2 is not None:
        if two_by_two_matrix1.shape != (2,2):
            raise Exception("Matrix #2 must be 2x2")
    
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
        color=matrix1_color,
        alpha=0.25
    )
    plt.fill(
        transformed_square[0,:],
        transformed_square[1,:],
        color=matrix1_color,
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
            color=matrix1_color
        )
    
    # If a second matrix is provided, plot the transformed unit square
    if two_by_two_matrix2 is not None:
        # Plot the transformed unit square
        plt.plot(
            transformed_square_2[0,:],
            transformed_square_2[1,:],
            color=matrix2_color,
            alpha=0.25
        )
        plt.fill(
            transformed_square_2[0,:],
            transformed_square_2[1,:],
            color=matrix2_color,
            alpha=0.1
        )
        # Add label for transformed unit square
        if show_labels:
            plt.text(
                transformed_square_2[0,2]+0.15,
                transformed_square_2[1,2]+0.15, 
                "Transformed Unit Square #2", 
                fontsize=8, 
                fontfamily='Arial', 
                ha='center', 
                va='center', 
                color=matrix2_color
            )
    
    # Add title
    plt.title("Linear Transformation", fontsize=20, fontfamily='Arial')
    
    # Show the plot
    plt.show()

