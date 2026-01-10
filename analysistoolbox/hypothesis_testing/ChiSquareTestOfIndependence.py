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
                                title_font_size=14,
                                subtitle_y_indent=1.1,
                                subtitle_font_size=11,
                                caption_y_indent=-0.15,
                                caption_font_size=8,
                                decimal_places_for_data_label=1):
    """
    Perform a chi-square test of independence between two categorical variables.

    This function conducts a chi-square test of independence to determine whether there is
    a statistically significant association between two categorical variables. The test
    compares observed frequencies in a contingency table with expected frequencies under
    the null hypothesis of independence (no association).

    The chi-square test of independence is essential for:
      * Determining if two categorical variables are related or independent
      * Analyzing survey data and cross-tabulations
      * Testing associations in medical and clinical research
      * Market research and customer segmentation analysis
      * Quality control and process improvement studies
      * Social science research on demographic patterns
      * A/B testing and experimental design validation

    The function calculates the chi-square statistic, p-value, and degrees of freedom,
    and provides both observed and expected frequency counts for each cell in the
    contingency table. Results can be visualized with a clustered bar chart comparing
    observed versus expected counts, making it easy to identify where associations
    deviate from independence.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the categorical variables to analyze.
    categorical_outcome_column
        Name of the column containing the categorical outcome variable (dependent variable).
        This will be displayed on the x-axis of the visualization.
    categorical_predictor_column
        Name of the column containing the categorical predictor variable (independent variable).
        This will be used to create separate facets in the visualization.
    show_contingency_tables
        If True, prints the observed and expected contingency tables to the console.
        Defaults to True.
    show_plot
        If True, displays a clustered bar chart comparing observed and expected counts
        for each combination of categories. Defaults to True.
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
        (observed minus expected) for each combination of the categorical outcome and
        predictor variables. Columns include the outcome variable, predictor variable,
        'Observed', 'Expected', and 'Difference (observed minus expected)'.

    Examples
    --------
    # Test independence between gender and product preference
    import pandas as pd
    df = pd.DataFrame({
        'gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Female'] * 20,
        'product': ['A', 'B', 'A', 'B', 'A', 'B'] * 20
    })
    results = ChiSquareTestOfIndependence(
        dataframe=df,
        categorical_outcome_column='product',
        categorical_predictor_column='gender'
    )

    # Test association between education level and voting preference
    voting_data = pd.DataFrame({
        'education': ['High School', 'College', 'Graduate'] * 50,
        'vote': ['Yes', 'No', 'Yes'] * 50
    })
    results = ChiSquareTestOfIndependence(
        dataframe=voting_data,
        categorical_outcome_column='vote',
        categorical_predictor_column='education',
        show_contingency_tables=True,
        show_plot=True
    )

    # Analyze customer satisfaction by region with custom styling
    satisfaction_df = pd.DataFrame({
        'region': ['North', 'South', 'East', 'West'] * 30,
        'satisfaction': ['Satisfied', 'Neutral', 'Dissatisfied'] * 40
    })
    results = ChiSquareTestOfIndependence(
        dataframe=satisfaction_df,
        categorical_outcome_column='satisfaction',
        categorical_predictor_column='region',
        color_palette='Set2',
        figure_size=(8, 6),
        title_for_plot='Customer Satisfaction by Region',
        subtitle_for_plot='Chi-Square Test Results',
        caption_for_plot='Analysis shows regional differences in satisfaction levels.',
        data_source_for_plot='Customer Survey 2024'
    )

    # Quick test without visualizations
    results = ChiSquareTestOfIndependence(
        dataframe=df,
        categorical_outcome_column='outcome',
        categorical_predictor_column='treatment',
        show_contingency_tables=False,
        show_plot=False
    )

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
        index=[categorical_outcome_column, categorical_predictor_column],
        columns='Value Type',
        values='Count')
    df_counts = df_counts.rename_axis(columns = None).reset_index()
    
    # Add observed minus expected column
    df_counts['Difference (observed minus expected)'] = df_counts['Observed'] - df_counts['Expected']
    
    # Return counts 
    return(df_counts)

