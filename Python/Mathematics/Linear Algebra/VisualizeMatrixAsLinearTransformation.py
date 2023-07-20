import matplotlib.pyplot as plt
import numpy as np

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
    """_summary_
    This function takes in two 2x2 matrices and plots the unit square and the transformed unit square. If a second matrix is provided, it will plot the transformed unit square twice, once for each matrix.
    
    Args:
        two_by_two_matrix1 (list or np.array): A 2x2 matrix. If a list is provided, it will be converted to a numpy array.
        two_by_two_matrix2 (list or np.array, optional): A 2x2 matrix. If a list is provided, it will be converted to a numpy array. Defaults to None.
        plot_with_grid (bool, optional): Whether or not to plot a grid on plot. Defaults to False.
        show_labels (bool, optional): Whether or not to show labels on plot. Defaults to True.
        matrix1_color (str, optional): Color of the first matrix. Defaults to '#033dfc'.
        matrix2_color (str, optional): Color of the second matrix. Defaults to '#03b1fc'.
        x_min (int, optional): The minimum x value to plot. Defaults to -5.
        x_max (int, optional): The maximum x value to plot. Defaults to 5.
        y_min (int, optional): The minimum y value to plot. Defaults to -5.
        y_max (int, optional): The maximum y value to plot. Defaults to 5.
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

  
# # Test the function
# VisualizeMatrixAsLinearTransformation([[3, 1],[1, 2]])
# # VisualizeMatrixAsLinearTransformation([[0, 1],[1, 1]])

# # theta = np.pi/3 # 60 degree clockwise rotation
# # a = np.column_stack([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
# # VisualizeMatrixAsLinearTransformation(a)

# # m_1 = np.array([[3, 1], [1, 2]])
# # m_2 = np.array([[2, -1], [0, 2]])
# # VisualizeMatrixAsLinearTransformation(
# #     two_by_two_matrix1=m_1,
# #     two_by_two_matrix2=m_2,
# #     x_min=-1,
# #     y_min=0,
# #     y_max=7
# # )

# # m_1 = np.array([[3, 1], [1, 2]])
# # m_1_inv = np.linalg.inv(m_1)
# # VisualizeMatrixAsLinearTransformation(
# #     two_by_two_matrix1=m_1,
# #     two_by_two_matrix2=m_1_inv,
# #     x_min=-1,
# #     y_min=0,
# #     y_max=7
# # )
