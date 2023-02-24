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
    
    # Return the dot_product
    return dot_product


# # Test the function
# CalculateDotProduct(
#     matrixA=[2, 4],
#     matrixB=[3, 1]
# )
