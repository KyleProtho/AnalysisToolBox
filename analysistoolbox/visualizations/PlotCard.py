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
    """
    Create a clean, card-style visualization for a single KPI or metric.

    This function generates a professional-looking "KPI card" using matplotlib. 
    It focuses on a single large numeric or text value, accompanied by an optional 
    descriptive label. The visualization is designed for dashboard-style summaries, 
    executive reports, or as a standalone graphic. It provides detailed control 
    over typography, colors, and figure dimensions.

    KPI cards are essential for:
      * Healthcare: Displaying current hospital occupancy rates or daily admissions.
      * Epidemiology: Highlighting the total number of active cases or vaccination milestones.
      * Intelligence Analysis: Showcasing the number of high-priority signal intercepts in a cycle.
      * Data Science: Displaying top-level model metrics like R-squared or AUC.
      * Finance: Highlighting total revenue or profit margin for a specific period.
      * Operations: Displaying current system uptime or number of processed units.
      * Marketing: Showcasing total conversion count or average customer acquisition cost.
      * Public Health: Highlighting the current air quality index (AQI) for a specific city.

    Parameters
    ----------
    value : str or int or float
        The primary data point to be displayed prominently in the center of the card.
    value_label : str, optional
        A descriptive label displayed below the primary value (e.g., "Total Active Cases"). 
        Defaults to None.
    value_font_color : str, optional
        The hex color code or name for the primary value text. Defaults to '#262626'.
    value_font_size : int, optional
        The font size for the primary value text in points. Defaults to 30.
    value_font_family : str, optional
        The font family name for the primary value (e.g., 'Arial', 'Roboto'). 
        Defaults to 'Arial'.
    value_font_weight : str, optional
        The font weight for the primary value (e.g., 'normal', 'bold', 'light'). 
        Defaults to 'bold'.
    value_label_font_color : str, optional
        The hex color code or name for the secondary label text. Defaults to '#595959'.
    value_label_font_size : int, optional
        The font size for the secondary label text in points. Defaults to 14.
    value_label_font_family : str, optional
        The font family name for the secondary label. Defaults to 'Arial'.
    value_label_font_weight : str, optional
        The font weight for the secondary label (e.g., 'normal', 'bold'). 
        Defaults to 'normal'.
    figure_size : tuple, optional
        The dimensions of the output figure as a (width, height) tuple in inches. 
        Defaults to (3, 2).
    filepath_to_save_plot : str, optional
        The local path (ending in .png or .jpg) where the card should be exported. 
        If None, the file is not saved. Defaults to None.

    Returns
    -------
    None
        The function displays the card using matplotlib and optionally saves 
        it to disk with a transparent background.

    Examples
    --------
    # Healthcare: Displaying daily patient admissions
    PlotCard(
        value=142,
        value_label="Daily Admissions",
        value_font_color="#b0170c",
        figure_size=(4, 2)
    )

    # Epidemiology: highlighting a vaccination milestone
    PlotCard(
        value="85.4%",
        value_label="Community Vaccination Rate",
        value_font_color="#2ecc71",
        value_font_weight="bold"
    )

    # Intelligence Analysis: Signal Intercept Count
    PlotCard(
        value=12,
        value_label="Priority-1 Signal Intercepts",
        value_font_color="#2c3e50",
        value_label_font_size=10
    )
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

