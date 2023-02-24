import matplotlib.pyplot as plt
import numpy as np

def VisualizeMatrixAsLinearTransformation(two_by_two_matrix):
    # If matrix is list, convert to numpy array
    if type(two_by_two_matrix) == list:
        two_by_two_matrix = np.array(two_by_two_matrix)
        
    # Ensure matrix is 2x2
    if two_by_two_matrix.shape != (2,2):
        raise Exception("Matrix must be 2x2")
    
    # Plot the vectors
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Set the x and y limits
    x_min = two_by_two_matrix[0,:].min()
    if x_min > 0:
        x_min = 0
    x_max = two_by_two_matrix[0,:].max()
    if x_max < 0:
        x_max = 0
    y_min = two_by_two_matrix[1,:].min()
    if y_min > 0:
        y_min = 0
    y_max = two_by_two_matrix[1,:].max()
    if y_max < 0:
        y_max = 0
    ax.set_xlim([x_min-2, x_max+2])
    ax.set_ylim([y_min-2, y_max+2])
    
    # Plot the unit square
    plt.plot([0,1,1,0,0],[0,0,1,1,0],
             color='black',
             alpha=0.5)
    
    # Plot the transformed unit square
    transformed_square = two_by_two_matrix @ np.array([[0,1,1,0,0],[0,0,1,1,0]])
    plt.plot(transformed_square[0,:],transformed_square[1,:],color='red')
    
    # Add a grid
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
    
    # Plot the basis vectors
    plt.quiver([0,0],[0,0],[1,0],[0,1],
               angles='xy',
               scale_units='xy',
               scale=1,
               color='black',
               alpha=0.5)
    
    # Plot the transformed basis vectors
    plt.quiver([0,0],[0,0],[two_by_two_matrix[0,0],two_by_two_matrix[0,1]],[two_by_two_matrix[1,0],two_by_two_matrix[1,1]],angles='xy',scale_units='xy',scale=1,color=['r','r'])

    
# # Test the function
# VisualizeMatrixAsLinearTransformation([[3, 1],[1, 2]])
# # theta = np.pi/3 # 60 degree clockwise rotation
# # a = np.column_stack([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
# # VisualizeMatrixAsLinearTransformation(a)
