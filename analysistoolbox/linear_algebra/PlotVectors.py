# Load packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Declare function
def PlotVectors(list_of_vectors,
                list_of_vector_labels=None,
                color_palette='Set2'):
    """
    Visualize a collection of 2D vectors on a coordinate plane.

    This function provides a clean, automated way to plot multiple 2D vectors using 
    Matplotlib and Seaborn. It handles vector conversion, coordinate scaling, and 
    aesthetic labeling, making it easy to compare the magnitude and direction of 
    multiple data points or forces simultaneously.

    Visualizing vectors is essential for:
      * Mapping movement patterns or displacement offsets
      * Visualizing feature embeddings or PCA components in 2D space
      * Representing the magnitude and direction of economic shifts
      * Modeling force vectors or directional threats in a perimeter
      * Explaining linear combinations and vector operations
      * Analyzing stress or load directions in structural simplifies

    The function supports custom Seaborn color palettes for professional styling and 
    automatically calculates appropriate axis limits to ensure all vectors are 
    visible and centered within the grid.

    Parameters
    ----------
    list_of_vectors
        A list of 2D vectors. Each vector can be represented as a list or a 
        numpy.ndarray (e.g., [x, y]).
    list_of_vector_labels
        A list of string labels corresponding to each vector. If None, default 
        labels (e.g., 'Vector 0', 'Vector 1') are generated. Defaults to None.
    color_palette
        The name of the Seaborn color palette to use for styling the vectors. 
        Defaults to 'Set2'.

    Returns
    -------
    None
        The function renders a plot directly to the active window using plt.show().

    Raises
    ------
    Exception
        If any provided vector has more than 2 coordinates.

    Examples
    --------
    # Plot basic unit vectors
    PlotVectors([[1, 0], [0, 1]], ['Unit X', 'Unit Y'])

    # Plot vectors representing data movement with a custom palette
    import numpy as np
    migration_paths = [np.array([2, 3]), np.array([-1, 4]), np.array([5, -2])]
    PlotVectors(
        migration_paths, 
        ['Region A', 'Region B', 'Region C'], 
        color_palette='viridis'
    )

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

