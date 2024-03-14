# Load packagesim
import matplotlib.pyplot as plt

# Declare function
def PlotCard(value,
             value_label=None,
             # Text formatting arguments
             value_font_color='#262626',
             value_font_size=30,
             value_font_family='Arial',
             value_font_weight='bold',
             # Value label formatting arguments
             value_label_font_color='#595959',
             value_label_font_size=14,
             value_label_font_family='Arial',
             value_label_font_weight='normal',
             # Plot formatting arguments
             figure_size=(3, 2),
             # Plot saving arguments
             filepath_to_save_plot=None):
    """Creates a simple card-like visualization with a value and an optional value label.

    Args:
        value: The main value to be displayed in the card.
        value_label: The label for the value (optional).
        value_font_color: The font color for the value (default: '#262626').
        value_font_size: The font size for the value (default: 30).
        value_font_family: The font family for the value (default: 'Arial').
        value_font_weight: The font weight for the value (default: 'bold').
        value_label_font_color: The font color for the value label (default: '#595959').
        value_label_font_size: The font size for the value label (default: 14).
        value_label_font_family: The font family for the value label (default: 'Arial').
        value_label_font_weight: The font weight for the value label (default: 'normal').
        figure_size: The size of the plot figure (default: (3, 2)).
        filepath_to_save_plot: The filepath to save the plot as an image file (optional).

    Returns:
        None
    """
    # Initialize figure
    f, ax = plt.subplots(figsize=figure_size)
    
    # Hide axes
    ax.axis('off')
    
    # Place value text
    if value_label is None:
        ax.text(
            x=0.5, 
            y=0.5, 
            s=f'{value}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=value_font_size,
            fontfamily=value_font_family,
            fontweight=value_font_weight,
            color=value_font_color
        )
    else:
        ax.text(
            x=0.5,
            y=0.6,
            s=f'{value}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=value_font_size,
            fontfamily=value_font_family,
            fontweight=value_font_weight,
            color=value_font_color
        )
        
    # Place value label text, if provided
    if value_label is not None:
        ax.text(
            x=0.5,
            y=0.4,
            s=f'{value_label}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=value_label_font_size,
            fontfamily=value_label_font_family,
            fontweight=value_label_font_weight,
            color=value_label_font_color
        )
        
    # If filepath_to_save_plot is provided, save the plot
    if filepath_to_save_plot != None:
        # Ensure that the filepath ends with '.png' or '.jpg'
        if not filepath_to_save_plot.endswith('.png') and not filepath_to_save_plot.endswith('.jpg'):
            raise ValueError("The filepath to save the plot must end with '.png' or '.jpg'.")
        
        # Save plot
        plt.savefig(
            filepath_to_save_plot, 
            bbox_inches="tight", 
            transparent=True
        )
        
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()

