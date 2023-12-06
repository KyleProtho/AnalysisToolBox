# Load packages
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import seaborn as sns
import textwrap

# Declare function
def ChiSquareTestOfIndependence(dataframe,
                                categorical_outcome_column,
                                categorical_predictor_column,
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
                                subtitle_y_indent=1.1,
                                caption_y_indent=-0.15,
                                decimal_places_for_data_label=1):
    """
    Performs a chi-square test of independence and returns the observed and expected counts for each combination of the outcome and predictor variables. 
    It also returns a clustered bar chart of the observed and expected counts.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data.
        categorical_outcome_column (str): The name of the outcome column.
        categorical_predictor_column (str): The name of the predictor column.
        show_contingency_tables (bool, optional): If True, prints the contingency tables. Defaults to True.
        show_plot (bool, optional): If True, shows a plot of the observed and expected counts. Defaults to True.
        color_palette (str, optional): The color palette to use for the plot. Defaults to "Set1".
        fill_transparency (float, optional): The transparency of the bars in the plot. Defaults to 0.8.
        figure_size (tuple, optional): The size of the figure for the plot. Defaults to (6, 6).
        title_for_plot (str, optional): The title for the plot. Defaults to "Chi-Square Test of Independence".
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to "Shows observed vs. expected counts".
        caption_for_plot (str, optional): The caption for the plot. Defaults to "Expected counts are based on the null hypothesis of no association between the two variables.".
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        x_indent (float, optional): The x-indent for the plot title. Defaults to -0.95.
        title_y_indent (float, optional): The y-indent for the plot title. Defaults to 1.15.
        subtitle_y_indent (float, optional): The y-indent for the plot subtitle. Defaults to 1.1.
        caption_y_indent (float, optional): The y-indent for the plot caption. Defaults to -0.15.
        decimal_places_for_data_label (int, optional): The number of decimal places for the data labels on the plot. Defaults to 1.

    Returns:
        pandas.DataFrame: A dataframe containing the observed and expected counts for each combination of the outcome and predictor variables, as well as the difference between the observed and expected counts.
    """
    
    # Select necessary columns
    dataframe = dataframe[[
        categorical_predictor_column,
        categorical_outcome_column
    ]]
    
    # Create contingency table in Python
    contingency_table = sm.stats.Table.from_data(dataframe)
    
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
        id_vars=categorical_predictor_column,
        var_name=categorical_outcome_column,
        value_name='Observed'
    )
    df_temp = pd.DataFrame(contingency_table.fittedvalues.round(2))
    df_temp = df_temp.reset_index()
    df_temp = df_temp.melt(
        id_vars=categorical_predictor_column,
        var_name=categorical_outcome_column,
        value_name='Expected'
    )
    df_counts = df_counts.merge(
        df_temp,
        how='left',
        on=[categorical_predictor_column, categorical_outcome_column]
    )
    
    # Convert to long-form
    df_counts = df_counts.melt(
        id_vars=[categorical_predictor_column, categorical_outcome_column],
        var_name='Value Type',
        value_name='Count'
    )
    
    # Show clustered bar chart of observed and expected values, if requested
    if show_plot:
        # Draw a nested barplot by species and sex
        ax = sns.catplot(
            data=df_counts,
            kind='bar',
            x=categorical_outcome_column,
            y='Count',
            hue='Value Type',
            col=categorical_predictor_column,
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
            wrapped_x_labels = "\n".join(ax.get_xlabel()[j:j+30] for j in range(0, len(ax.get_xlabel()), 30))
            ax.set_xlabel(wrapped_x_labels, horizontalalignment='center')
        
            # Format and wrap x axis tick labels using textwrap
            x_tick_labels = ax.get_xticklabels()
            wrapped_x_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 30)) for label in x_tick_labels]
            ax.set_xticklabels(wrapped_x_tick_labels, fontsize=10, fontname="Arial", color="#262626")
            
             # Word wrap the column labels without splitting words
            wrapped_column_labels = textwrap.fill(ax.get_title(), 60, break_long_words=False)
            ax.set_title(wrapped_column_labels, fontname="Arial", fontsize=10, color="#262626")
            
            # Change x-axis colors to "#666666"
            ax.tick_params(axis='x', colors="#666666")
            ax.spines['top'].set_color("#666666")
            ax.spines['left'].set_color("#d6d6d6")
        
            # Remove bottom, left, and right spines
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
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
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            fontname="Arial",
            fontsize=11,
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
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
        
        # Show plot
        plt.show()
        
        # Clear plot
        plt.clf()
    
    # Pivot value type column before return obs vs exp counts
    df_counts = df_counts.pivot(
        index=[categorical_outcome_column, categorical_predictor_column],
        columns='Value Type',
        values='Count')
    df_counts = df_counts.rename_axis(columns = None).reset_index()
    
    # Add observed minus expected column
    df_counts['Difference (observed minus expected)'] = df_counts['Observed'] - df_counts['Expected']
    
    # Return counts 
    return(df_counts)

