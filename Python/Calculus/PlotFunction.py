import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

# Define a function that plots a function
def PlotFunction(f_of_x, 
                 minimum_x=-10, 
                 maximum_x=10, 
                 n=100):
    x = np.linspace(minimum_x, maximum_x, n)
    y = f_of_x(x)
    plt.plot(x, y)
    
    # Show the plot
    plt.show()

# # Test the function
# PlotFunction(
#     f_of_x=lambda x: x**2
# )
