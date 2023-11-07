# Load packages
import os

# Declare function
def CreateFileTree(path, 
                   indent_spaces=2):
    """
    Recursively walks the directory tree and prints a diagram of all the subdirectories and files in the tree.

    Args:
        path (str): The path to the root directory.
        indent (str): The number of spaces to indent the output.
    """
    
    # Initialize the output
    output = []
    
    # Walk the directory tree
    for root, dirs, files in os.walk(path):
        # Get the current directory
        for dir in dirs:
            output.append(f'{dir}')
            # Get the files in the current directory
            for file in os.listdir(os.path.join(root, dir)):
                # Indent the files
                output.append(f'├{"─" * indent_spaces}{file}')

    # Print the output
    print('\n'.join(output))

