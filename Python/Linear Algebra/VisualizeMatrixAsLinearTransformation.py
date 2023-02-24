import matplotlib.pyplot as plt
import numpy as np

def VisualizeMatrixAsLinearTransformation(two_by_two_matrix1,
                                        #   two_by_two_matrix2=None,
                                          plot_with_grid=False):
    # If matrix is list, convert to numpy array
    if type(two_by_two_matrix1) == list:
        two_by_two_matrix1 = np.array(two_by_two_matrix1)
    # if two_by_two_matrix2 is not None:
    #     if type(two_by_two_matrix2) == list:
    #         two_by_two_matrix2 = np.array(two_by_two_matrix2)
            
    # Perform matrix multiplication
    transformed_square = np.dot(two_by_two_matrix1, np.array([[0,1,1,0,0],[0,0,1,1,0]]))
    # if two_by_two_matrix2 is not None:
    #     transformed_square_2 = np.dot(two_by_two_matrix2, two_by_two_matrix1)
        
    # Ensure matrix is 2x2
    if two_by_two_matrix1.shape != (2,2):
        raise Exception("Matrix #1 must be 2x2")
    # if two_by_two_matrix2 is not None:
    #     if two_by_two_matrix1.shape != (2,2):
    #         raise Exception("Matrix #2 must be 2x2")
    
    # Plot the vectors
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Set the x and y limits
    # if two_by_two_matrix2 is None:
    #     x_min = min([0, transformed_square[0,:].min()])
    #     x_max = max([0, transformed_square[0,:].max()])
    #     y_min = min([0, transformed_square[1,:].min()])
    #     y_max = max([0, transformed_square[1,:].max()])
    # else:
    #     x_min = min([0, transformed_square[0,:].min(), transformed_square_2[0,:].min()])
    #     x_max = max([0, transformed_square[0,:].max(), transformed_square_2[0,:].max()])
    #     y_min = min([0, transformed_square[1,:].min(), transformed_square_2[1,:].min()])
    #     y_max = max([0, transformed_square[1,:].max(), transformed_square_2[1,:].max()])
    x_min = min([0, transformed_square[0,:].min()])
    x_max = max([0, transformed_square[0,:].max()])
    y_min = min([0, transformed_square[1,:].min()])
    y_max = max([0, transformed_square[1,:].max()])
    ax.set_xlim([x_min-1, x_max+1])
    ax.set_ylim([y_min-1, y_max+1])
    
    # Add a grid
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
    
    # Plot the transformed unit square
    plt.plot(transformed_square[0,:],transformed_square[1,:],color='red')
    
    # Plot the basis vectors
    plt.quiver([0,0],[0,0],[1,0],[0,1],
               angles='xy',
               scale_units='xy',
               scale=1,
               color='black',
               alpha=0.5)
    
    # Plot the transformed basis vectors
    plt.quiver(
        [0,0],
        [0,0],
        two_by_two_matrix1[0,:],
        two_by_two_matrix1[1,:],
        angles='xy',
        scale_units='xy',
        scale=1,
        color=['r','r']
    )
    
    # # If a second matrix is provided, plot the transformed unit square
    # if two_by_two_matrix2 is not None:        
    #     # Plot the transformed square
    #     plt.plot(
    #         transformed_square_2[0,:],
    #         transformed_square_2[1,:],
    #         color='blue'
    #     )
        
    #     # Plot the second transformed basis vectors
    #     print(two_by_two_matrix1[0,:])
    #     print(two_by_two_matrix1[0,:] + two_by_two_matrix2[0,:])
    #     plt.quiver(
    #         two_by_two_matrix1[1,:],
    #         two_by_two_matrix1[0,:],
    #         # two_by_two_matrix2[1,:],
    #         # two_by_two_matrix2[0,:],
    #         angles='xy',
    #         scale_units='xy',
    #         scale=1,
    #         color=['b','b']
    #     )
    
    # Show the plot
    plt.show()
    
# Test the function
VisualizeMatrixAsLinearTransformation([[3, 1],[1, 2]])
# VisualizeMatrixAsLinearTransformation([[0, 1],[1, 1]])

# # theta = np.pi/3 # 60 degree clockwise rotation
# # a = np.column_stack([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
# # VisualizeMatrixAsLinearTransformation(a)

