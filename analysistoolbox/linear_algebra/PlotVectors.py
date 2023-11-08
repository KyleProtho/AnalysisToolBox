# Load packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Declare function
def PlotVectors(list_of_vectors,
                list_of_vector_labels=None,
                color_palette='Set2'):
    """
    Plots a list of 2D vectors.

    Args:
        list_of_vectors (list): A list of 2D vectors represented as lists or numpy arrays.
        list_of_vector_labels (list, optional): A list of labels for the vectors. If None, default labels will be used. Defaults to None.
        color_palette (str, optional): The name of the seaborn color palette to use. Defaults to 'Set2'.

    Raises:
        Exception: If a vector has more than 2 coordinates.

    Returns:
        None
    """
    
    # Iterate through the list of vectors, and if the vector is a list, convert it to a numpy array
    for vector_index in range(len(list_of_vectors)):
        # If the vector is a list, convert it to a numpy array
        if type(list_of_vectors[vector_index]) == list:
            list_of_vectors[vector_index] = np.array(list_of_vectors[vector_index])
        # If the vector has more than 2 coordinates, raise an exception
        if len(list_of_vectors[vector_index].shape) > 2:
            raise Exception('Vector {} has more than 2 coordinates'.format(vector_index))
    
    # If list_of_vector_labels is None, generate a list of labels
    if list_of_vector_labels is None:
        list_of_vector_labels= []
        for vector_index in range(len(list_of_vectors)):
            list_of_vector_labels.append('Vector {}'.format(vector_index))
    
    # Set the color palette
    sns.set_palette(color_palette)
    
    # Plot the vectors
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Set the x and y limits
    for vector in list_of_vectors:
        print(vector.min())
    x_min = min([0] + [vector.min() for vector in list_of_vectors])
    x_max = max([0] + [vector.max() for vector in list_of_vectors])
    y_min = min([0] + [vector.min()  for vector in list_of_vectors])
    y_max = max([0] + [vector.max()  for vector in list_of_vectors])
    ax.set_xlim([x_min-1, x_max+1])
    ax.set_ylim([y_min-1, y_max+1])
        
    # Plot each vector
    for i, v in enumerate(list_of_vectors):
        sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(v)])
        plt.quiver(
            v[0], 
            v[1],
            color=sns.color_palette()[i],
            angles='xy', 
            scale_units='xy', 
            scale=1
        )
        ax.text(
            v[0]-0.2+sgn[0],
            v[1]-0.2+sgn[1], 
            list_of_vector_labels[i],
            fontsize=10,
            color=sns.color_palette()[i]
        )
    
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
    
    # Show the plot
    plt.show()

