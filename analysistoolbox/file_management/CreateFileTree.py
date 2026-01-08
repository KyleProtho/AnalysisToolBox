# Load packages
import os

# Declare function
def CreateFileTree(path, 
                   indent_spaces=2):
    """
    Generate and print a visual tree diagram of directory structure.

    This function recursively traverses a directory tree and prints a hierarchical diagram
    showing all subdirectories and files using ASCII tree characters. It walks through the
    entire directory structure starting from the specified root path, displaying each directory
    and its contents with visual indentation using box-drawing characters (├ and ─). The output
    provides a clear, readable representation of the file system hierarchy, making it easy to
    understand project structure and organization at a glance.

    Directory tree visualization is essential for:
      * Documenting project structure in README files and documentation
      * Understanding unfamiliar codebases and directory layouts
      * Verifying correct file organization after restructuring
      * Creating visual guides for onboarding new team members
      * Debugging file path issues and missing directories
      * Presenting project organization in reports and presentations
      * Auditing directory contents and file placement
      * Planning directory reorganization and cleanup efforts

    The function uses Python's os.walk() for recursive traversal, automatically handling
    nested directories of any depth. It formats the output with customizable indentation
    using box-drawing characters to create a professional tree diagram. The visualization
    is printed directly to stdout, making it suitable for terminal output, logging, or
    capture for documentation purposes.

    Parameters
    ----------
    path
        Absolute or relative path to the root directory to visualize. The function will
        recursively traverse all subdirectories starting from this location.
    indent_spaces
        Number of horizontal dash characters (─) to use for indentation at each level,
        controlling the visual spacing of the tree diagram. Defaults to 2.

    Returns
    -------
    None
        This function does not return a value. It prints the directory tree diagram
        directly to stdout.

    Notes
    -----
    * **Recursive Traversal**: Processes all subdirectories regardless of depth
    * **Visual Format**: Uses ├ and ─ characters for tree structure visualization
    * **Output Destination**: Prints to stdout; redirect to capture in files
    * **Hidden Files**: Includes hidden files and directories (those starting with .)
    * **Permissions**: May skip directories without read permissions

    Examples
    --------
    # Visualize current project structure
    CreateFileTree('.')
    # Prints tree diagram of current directory with 2-space indentation

    # Visualize specific directory with custom indentation
    CreateFileTree(
        path='/Users/username/projects/myapp',
        indent_spaces=4
    )
    # Displays directory tree with 4-dash indentation

    # Document project structure
    CreateFileTree('src')
    # Example output:
    # components
    # ├──Header.js
    # ├──Footer.js
    # utils
    # ├──helpers.js
    # ├──constants.js

    # Capture tree output to file
    import sys
    from io import StringIO
    
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    CreateFileTree('project_root')
    tree_output = sys.stdout.getvalue()
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Save to file
    with open('directory_structure.txt', 'w') as f:
        f.write(tree_output)
    # Saves tree diagram to text file for documentation

    # Compare directory structures
    print("Development Structure:")
    CreateFileTree('dev_environment')
    print("\nProduction Structure:")
    CreateFileTree('prod_environment')
    # Displays both structures for comparison

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

