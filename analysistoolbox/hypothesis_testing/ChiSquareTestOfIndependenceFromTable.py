# Load packages
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import seaborn as sns
import textwrap

# Declare function
def ChiSquareTestOfIndependenceFromTable(contingency_table,
                                         column_wise_variable_name,
                                         show_contingency_tables=True,
                                         show_plot=True,
                                         # Plot formatting arguments
                                         color_palette="Paired",
                                         fill_transparency=0.8,
                                         figure_size=(6, 8),
                                         # Text formatting arguments
                                         title_for_plot="Chi-Square Test of Independence",
                                         subtitle_for_plot="Shows observed vs. expected counts",
                                         caption_for_plot="Expected counts are based on the null hypothesis of no association between the two variables.",
                                         data_source_for_plot=None,
                                         x_indent=-0.95,
                                         title_y_indent=1.15,
                                         title_font_size=14,
                                         subtitle_y_indent=1.1,
                                         subtitle_font_size=11,
                                         caption_y_indent=-0.15,
                                         caption_font_size=8,
                                         decimal_places_for_data_label=1):
    """
    Perform a chi-square test of independence from a pre-calculated contingency table.

    This function conducts a chi-square test of independence using a summarized contingency
    table rather than raw data. It determines whether there is a statistically significant
    association between two categorical variables by comparing the observed frequencies
    provided in the table with the frequencies that would be expected under the null
    hypothesis of independence.

    The chi-square test of independence from a table is essential for:
      * Analyzing research results presented in summary format
      * Validating published findings from cross-tabulations
      * Comparing experimental results against theoretical distributions
      * Assessing associations in historical data where raw records are unavailable
      * Evaluating market share distributions across different segments
      * Performing meta-analysis on aggregated categorical data
      * Conducting quick diagnostic tests on frequency counts

    The function calculates the chi-square statistic, p-value, and degrees of freedom
    directly from the input table. It also provides a visualization via a clustered
    bar chart that compares observed versus expected counts. This is particularly
    useful for identifying specific cells in the table where the association deviates 
    most significantly from what would be expected by chance.

    Parameters
    ----------
    contingency_table
        A pandas DataFrame representing the contingency table. The rows should represent
        one categorical variable (accessible via the index) and the columns should
        represent another. The values must be non-negative frequency counts.
    column_wise_variable_name
        A descriptive name for the categorical variable represented by the columns
        of the contingency table. This name will be used in the visualization and results.
    show_contingency_tables
        If True, prints the observed and expected contingency tables to the console.
        Defaults to True.
    show_plot
        If True, displays a clustered bar chart comparing observed and expected counts.
        Defaults to True.
    color_palette
        Seaborn color palette name to use for the plot bars. Defaults to 'Paired'.
    fill_transparency
        Transparency level for the bars in the plot, ranging from 0 (transparent) to 1
        (opaque). Defaults to 0.8.
    figure_size
        Tuple specifying the (width, height) of each subplot in inches. Defaults to (6, 8).
    title_for_plot
        Main title text to display at the top of the plot. Defaults to
        'Chi-Square Test of Independence'.
    subtitle_for_plot
        Subtitle text to display below the main title. Defaults to
        'Shows observed vs. expected counts'.
    caption_for_plot
        Caption text to display at the bottom of the plot explaining the visualization.
        Defaults to 'Expected counts are based on the null hypothesis of no association
        between the two variables.'.
    data_source_for_plot
        Optional text identifying the data source, displayed in the caption area.
        If None, no data source is shown. Defaults to None.
    x_indent
        Horizontal position for left-aligning title, subtitle, and caption text.
        Defaults to -0.95.
    title_y_indent
        Vertical position for the main title relative to the plot. Defaults to 1.15.
    title_font_size
        Font size in points for the main title. Defaults to 14.
    subtitle_y_indent
        Vertical position for the subtitle relative to the plot. Defaults to 1.1.
    subtitle_font_size
        Font size in points for the subtitle. Defaults to 11.
    caption_y_indent
        Vertical position for the caption relative to the plot. Defaults to -0.15.
    caption_font_size
        Font size in points for the caption text. Defaults to 8.
    decimal_places_for_data_label
        Number of decimal places to display in the bar chart data labels. Defaults to 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the observed counts, expected counts, and the difference
        (observed minus expected) for each combination of categories. The resulting
        DataFrame includes columns for both categorical variables, 'Observed',
        'Expected', and 'Difference (observed minus expected)'.

    Examples
    --------
    # Create a contingency table for hair color and eye color
    import pandas as pd
    data = {
        'Blue Eyes': [10, 15, 5],
        'Brown Eyes': [12, 10, 25]
    }
    table = pd.DataFrame(data, index=['Blonde Hair', 'Brown Hair', 'Black Hair'])
    table.index.name = 'Hair Color'

    # Perform the test
    results = ChiSquareTestOfIndependenceFromTable(
        contingency_table=table,
        column_wise_variable_name='Eye Color'
    )

    # Analyze survey results for transport preference by city
    survey_data = {
        'Public Transit': [150, 200],
        'Private Vehicle': [100, 50]
    }
    cities = pd.DataFrame(survey_data, index=['New York', 'Los Angeles'])
    cities.index.name = 'City'

    results = ChiSquareTestOfIndependenceFromTable(
        contingency_table=cities,
        column_wise_variable_name='Transport Mode',
        color_palette='viridis',
        title_for_plot='Transport Preference by City',
        data_source_for_plot='2023 Urban Mobility Study'
    )

    # Test without visualization for quick calculation
    results = ChiSquareTestOfIndependenceFromTable(
        contingency_table=table,
        column_wise_variable_name='Eye Color',
        show_contingency_tables=False,
        show_plot=False
    )

    """
    
    # Get row-wise variable names
    row_wise_variable_name = contingency_table.index.name
    
    # Create contingency table in Python
    contingency_table = sm.stats.Table(contingency_table)
    
    # Perform chi-square test of independence
    results = chi2_contingency(contingency_table.table_orig)
    
    # Print results of test
    print("Chi-square test statistic: " + str(round(results[0], 3)))
    print("p-value: " + str(round(results[1], 3)))
    print("Degrees of freedom: " + str(round(results[2], 3)))
    
    # Show contingency table with margins
    if show_contingency_tables:
        print("")
        print("Observed:")
        print(contingency_table.table_orig)
        print("")
        print("Expected:")
        print(contingency_table.fittedvalues.round(2))
        
    # Combine observed and expected counts
    df_counts = pd.DataFrame(contingency_table.table_orig)
    df_counts = df_counts.reset_index()
    df_counts = df_counts.melt(
        id_vars=row_wise_variable_name,
        var_name=column_wise_variable_name,
        value_name='Observed'
    )
    df_temp = pd.DataFrame(contingency_table.fittedvalues.round(2))
    df_temp = df_temp.reset_index()
    df_temp = df_temp.melt(
        id_vars=row_wise_variable_name,
        var_name=column_wise_variable_name,
        value_name='Expected'
    )
    df_counts = df_counts.merge(
        df_temp,
        how='left',
        on=[row_wise_variable_name, column_wise_variable_name]
    )
    
    # Convert to long-form
    df_counts = df_counts.melt(
        id_vars=[row_wise_variable_name, column_wise_variable_name],
        var_name='Value Type',
        value_name='Count'
    )
    
    # Show clustered bar chart of observed and expected values, if requested
    if show_plot:
        # Draw a nested barplot by species and sex
        ax = sns.catplot(
            data=df_counts,
            kind='bar',
            x=column_wise_variable_name,
            y='Count',
            hue='Value Type',
            col=row_wise_variable_name,
            palette=color_palette, 
            alpha=fill_transparency, 
            height=figure_size[1],
            aspect=figure_size[0] / figure_size[1],
            ci=None,
        )
        
        # Add space between the title and the plot
        plt.subplots_adjust(top=0.85)
        
        # Iterate through subplots
        for ax in ax.axes.flat:
            # Wrap x axis label using textwrap
            wrapped_x_labels = textwrap.fill(ax.get_xlabel(), 60, break_long_words=False)
            ax.set_xlabel(wrapped_x_labels, fontsize=10, fontname="Arial", color="#262626", horizontalalignment='center')
        
            # Format and wrap x axis tick labels using textwrap
            x_tick_labels = ax.get_xticklabels()
            wrapped_x_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 30, break_long_words=False)) for label in x_tick_labels]
            ax.set_xticklabels(wrapped_x_tick_labels, fontsize=10, fontname="Arial", color="#262626")
            
             # Word wrap the column labels without splitting words
            wrapped_column_labels = textwrap.fill(ax.get_title(), 60, break_long_words=False)
            ax.set_title(wrapped_column_labels, fontname="Arial", fontsize=10, color="#262626", horizontalalignment='center')
            
            # Change x-axis colors to "#666666"
            ax.tick_params(axis='x', colors="#666666")
            ax.spines['top'].set_color("#666666")
        
            # Remove bottom, left, and right spines
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
            # Remove the y-axis tick text
            ax.set_yticklabels([])
        
            # Add data labels to each bar in the plot
            # Get minimum value absolute value of the data
            min_value = df_counts['Count'].abs().min()
            if min_value < 1:
                min_value = 1
            
            # Iterate through each bar in the plot and add a data label
            for p in ax.patches:
                # Get the height of each bar
                height = p.get_height()
                
                # Format the height to the specified number of decimal places
                height = round(height, decimal_places_for_data_label)
                
                # Add the data label to the plot, using the specified number of decimal places
                value_label = "{:,.{decimal_places}f}".format(height, decimal_places=decimal_places_for_data_label)
                ax.text(
                    x=p.get_x() + p.get_width() / 2.,
                    y=height + (min_value * 0.05),
                    s=value_label,
                    ha="center",
                    fontname="Arial",
                    fontsize=10,
                    color="#262626"
                )
            
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent,
            y=title_y_indent,
            s=title_for_plot,
            fontname="Arial",
            fontsize=title_font_size,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            fontname="Arial",
            fontsize=subtitle_font_size,
            color="#666666",
            transform=ax.transAxes
        )
        
            
        # Add a word-wrapped caption if one is provided
        if caption_for_plot != None or data_source_for_plot != None:
            # Create starting point for caption
            wrapped_caption = ""
            
            # Add the caption to the plot, if one is provided
            if caption_for_plot != None:
                # Word wrap the caption without splitting words
                wrapped_caption = textwrap.fill(caption_for_plot, 110, break_long_words=False)
                
            # Add the data source to the caption, if one is provided
            if data_source_for_plot != None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            
            # Add the caption to the plot
            ax.text(
                x=x_indent,
                y=caption_y_indent,
                s=wrapped_caption,
                fontname="Arial",
                fontsize=caption_font_size,
                color="#666666",
                transform=ax.transAxes
            )
        
        # Show plot
        plt.show()
        
        # Clear plot
        plt.clf()
    
    
    # Pivot value type column before return obs vs exp counts
    df_counts = df_counts.pivot(
        index=[column_wise_variable_name, row_wise_variable_name],
        columns='Value Type',
        values='Count')
    df_counts = df_counts.rename_axis(columns = None).reset_index()
    
    # Add observed minus expected column
    df_counts['Difference (observed minus expected)'] = df_counts['Observed'] - df_counts['Expected']
    
    # Return counts 
    return(df_counts)

