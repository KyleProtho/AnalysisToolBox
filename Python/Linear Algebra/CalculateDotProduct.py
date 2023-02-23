import matplotlib.pyplot as plt
import numpy as np

def CalculateDotProduct(matrixA, 
                        matrixB,
                        plot_dot_product=True):
    # If matrixA is a list, convert it to a numpy array
    if type(matrixA) == list:
        matrixA = np.array(matrixA)
    
    # If matrixB is a list, convert it to a numpy array
    if type(matrixB) == list:
        matrixB = np.array(matrixB)
    
    # Calculate the dot product of matrixA and matrixB
    dot_product = np.dot(matrixA, matrixB)
    
    # If plot_dot_product is True and matrixA and matrixB are 2D, plot the dot product
    if plot_dot_product: 
        if matrixA.shape == (2,) and matrixB.shape == (2,):
            # Plot the matrixA as a line with arrow head
            plt.quiver([0, 0], [0, 0], matrixA[0], matrixA[1], 
                       angles='xy', scale_units='xy', scale=1, color=['r', 'b'])
            
            # Plot the matrixB as a line with arrow head
            plt.quiver([0, 0], [0, 0], matrixB[0], matrixB[1], 
                       angles='xy', scale_units='xy', scale=1, color=['g', 'y'])
             
            # Set the x and y limits
            x_max = max(matrixA[0], matrixB[0], dot_product)
            y_max = max(matrixA[1], matrixB[1], dot_product)
            plt.xlim(x_max*-1, x_max*1.10)
            plt.ylim(y_max*-1, y_max*1.10)
            
            # Remove borders
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            
            # Add line at x=0 and y=0
            plt.axhline(y=0, color='k')
            plt.axvline(x=0, color='k')
            
            # Show plot
            plt.show()
    
    # Return the dot_product
    return dot_product


# # Test the function
# CalculateDotProduct(
#     matrixA=[2, 4],
#     matrixB=[3, 1]
# )
