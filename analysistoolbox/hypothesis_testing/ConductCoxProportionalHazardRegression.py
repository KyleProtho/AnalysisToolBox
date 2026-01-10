# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import textwrap

# Declare function
def ConductCoxProportionalHazardRegression(dataframe,
                                           outcome_column,
                                           duration_column,
                                           list_of_predictor_columns,
                                           plot_survival_curve=True,
                                           line_color="#3269a8",
                                           line_alpha=0.8,
                                           color_palette="Set2",
                                           markers="o",
                                           number_of_x_axis_ticks=20,
                                           x_axis_tick_rotation=None,
                                           title_for_plot="Survival Counts by Group",
                                           subtitle_for_plot='Shows the counts of "surviving" records by group over time',
                                           caption_for_plot=None,
                                           data_source_for_plot=None,
                                           x_indent=-0.127,
                                           title_y_indent=1.125,
                                           subtitle_y_indent=1.05,
                                           caption_y_indent=-0.3,
                                           figure_size=(8, 5)):
    """
    Conduct Cox Proportional Hazard Regression to model survival times and analyze predictor impacts.

    This function performs a Cox Proportional Hazard Regression, a semi-parametric
    survival analysis method used to explore the relationship between the survival
    time of objects and one or more predictor variables. It models the hazard rate,
    allowing for the estimation of how various factors influence the risk of an
    event occurring over time.

    Cox Proportional Hazard Regression is essential for:
      * Predicting customer churn and identifying high-risk segments
      * Analyzing patient survival rates in clinical and medical research
      * Modeling time-to-failure in industrial reliability engineering
      * Understanding factors influencing employee tenure and retention
      * Assessing credit risk and time until loan default or delinquency
      * Evaluating time on market for real estate or e-commerce listings
      * Optimizing marketing interventions based on expected duration till next action

    The function fits the regression model using the `lifelines` library and provides
    a statistical summary of coefficients, hazard ratios, and p-values. It also 
    includes an optional visualization that tracks the count of "surviving" records
    within different outcome groups over time, facilitating a quick visual 
    assessment of survival patterns before and after modeling.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the survival data, including outcome,
        duration, and predictor status.
    outcome_column
        Name of the column containing the binary event status (e.g., 1 for event,
        0 for censored) or categorical outcome.
    duration_column
        Name of the column containing the time-to-event or duration until the
        observation ended.
    list_of_predictor_columns
        A list of column names for the independent variables (covariates) to
        include in the regression model.
    plot_survival_curve
        If True, displays a visualization showing the count of surviving records
        over time for each group. Defaults to True.
    line_color
        The primary color for lines in the survival plot when hue is not applied.
        Defaults to "#3269a8".
    line_alpha
        The transparency level for the lines in the plot, ranging from 0 to 1.
        Defaults to 0.8.
    color_palette
        The seaborn color palette to use for distinguishing survival groups in
        the plot. Defaults to "Set2".
    markers
        The marker style to use for data points on the survival lines (e.g., 'o',
        's', 'x'). Defaults to "o".
    number_of_x_axis_ticks
        The number of major ticks to display on the x-axis (duration). Defaults to 20.
    x_axis_tick_rotation
        The rotation angle (in degrees) for the x-axis tick labels. Defaults to None.
    title_for_plot
        The main title text to display at the top of the plot. Defaults to
        "Survival Counts by Group".
    subtitle_for_plot
        The subtitle text to display below the main title. Defaults to
        'Shows the counts of "surviving" records by group over time'.
    caption_for_plot
        Caption text displayed at the bottom of the plot. Defaults to None.
    data_source_for_plot
        Optional text identifying the data source, displayed in the caption area.
        Defaults to None.
    x_indent
        Horizontal position for left-aligning the title, subtitle, and caption.
        Defaults to -0.127.
    title_y_indent
        Vertical position for the main title relative to the axes. Defaults to 1.125.
    subtitle_y_indent
        Vertical position for the subtitle relative to the axes. Defaults to 1.05.
    caption_y_indent
        Vertical position for the caption relative to the axes. Defaults to -0.3.
    figure_size
        Tuple specifying the (width, height) of the figure in inches. Defaults to (8, 5).

    Returns
    -------
    lifelines.CoxPHFitter
        The fitted Cox Proportional Hazard Regression model object, providing access
        to summary statistics, hazard ratios, and prediction methods.

    Examples
    --------
    # Analyze customer churn based on age and subscription type
    import pandas as pd
    df = pd.DataFrame({
        'churn': [1, 0, 1, 1, 0, 1] * 10,
        'tenure': [5, 12, 3, 8, 20, 2] * 10,
        'age': [25, 45, 32, 54, 21, 38] * 10,
        'is_premium': [1, 1, 0, 0, 1, 0] * 10
    })
    model = ConductCoxProportionalHazardRegression(
        dataframe=df,
        outcome_column='churn',
        duration_column='tenure',
        list_of_predictor_columns=['age', 'is_premium']
    )

    # Medical research: time to recovery with treatment groups
    medical_df = pd.DataFrame({
        'recovered': [1, 1, 0, 1] * 25,
        'days': [10, 15, 30, 20] * 25,
        'treatment_dose': [100, 200, 100, 200] * 25,
        'age': [40, 50, 60, 70] * 25
    })
    model = ConductCoxProportionalHazardRegression(
        dataframe=medical_df,
        outcome_column='recovered',
        duration_column='days',
        list_of_predictor_columns=['treatment_dose', 'age'],
        title_for_plot='Recovery Patterns by Treatment Status',
        color_palette='viridis'
    )

    # Industrial reliability test without plotting
    failure_df = pd.DataFrame({
        'failed': [1, 1, 1] * 20,
        'hours': [1000, 1500, 1200] * 20,
        'temperature': [70, 85, 90] * 20,
        'vibration': [0.5, 0.8, 1.2] * 20
    })
    model = ConductCoxProportionalHazardRegression(
        dataframe=failure_df,
        outcome_column='failed',
        duration_column='hours',
        list_of_predictor_columns=['temperature', 'vibration'],
        plot_survival_curve=False
    )

    """
    # Lazy load uncommon packages
    from lifelines import CoxPHFitter
    
    # Select the columns to keep
    dataframe = dataframe[[outcome_column, duration_column] + list_of_predictor_columns].copy()

    # Remove records with missing values
    dataframe = dataframe.dropna()

    # Order the dataframe by outcome
    dataframe = dataframe.sort_values(by = [outcome_column, duration_column], ascending = [True, True])

    # Create a new column showing unqiue status as group starting with 0
    unique_status = dataframe[outcome_column].unique()
    dataframe["Survival Group"] = dataframe[outcome_column].apply(lambda x: list(unique_status).index(x))
    # Print the survival group and their corresponding status
    print("Outcome variables have been assigned accordingly to each group:")
    print(dataframe[["Survival Group", outcome_column]].drop_duplicates().reset_index(drop = True))
    # Drop the survival group column
    dataframe = dataframe.drop(columns = [outcome_column])
    
    # Create a time series dataframe using the duration column
    # Get the maximum duration
    max_duration = dataframe[duration_column].max()
    # Create a time series dataframe
    time_series_dataframe = pd.DataFrame(
        data = range(0, max_duration + 1),
        columns = [duration_column]
    )
    
    # Iterate through the status groups
    for status_group in dataframe["Survival Group"].unique():
        # Create temporary copy of time series dataframe
        temp_time_series_dataframe = time_series_dataframe.copy()
        
        # Add the status group to the temporary time series dataframe
        temp_time_series_dataframe["Survival Group"] = status_group
        
        # Filter the original dataframe to the status group
        df_temp = dataframe[dataframe["Survival Group"] == status_group].copy()
        
        # Get the number of records in the dataframe that have a duration less than or equal to the time series dataframe
        temp_time_series_dataframe["Survival Count"] = temp_time_series_dataframe[duration_column].apply(
            lambda x: len(df_temp[df_temp[duration_column] > x])
        )
        
        # Row bind the temporary time series dataframe to the time series dataframe
        time_series_dataframe = pd.concat([time_series_dataframe, temp_time_series_dataframe], axis=0)
    # Filter out records with no survival group
    time_series_dataframe = time_series_dataframe[time_series_dataframe["Survival Group"].notnull()].reset_index(drop=True)
    
    # Use the time series dataframe to plot the survival curve
    if plot_survival_curve:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
    
        # Use Seaborn to create a line plot
        sns.lineplot(
            data=time_series_dataframe,
            x=duration_column,
            y="Survival Count",
            hue="Survival Group",
            palette=color_palette,
            alpha=line_alpha,
            marker=markers,
            ax=ax
        )
        
        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
    
        # Set the number of ticks on the x-axis
        if number_of_x_axis_ticks is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(number_of_x_axis_ticks))
            
        # Rotate x-axis tick labels
        if x_axis_tick_rotation is not None:
            plt.xticks(rotation=x_axis_tick_rotation)

        # Format tick labels to be Arial, size 9, and color #666666
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
        )
        
        # Format tick labels to be Arial, size 9, and color #666666
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
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
        
        # Word wrap the subtitle without splitting words
        if subtitle_for_plot != None:   
            subtitle_for_plot = textwrap.fill(subtitle_for_plot, 100, break_long_words=False)
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
        
        # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
        ax.yaxis.set_label_coords(-0.1, 0.84)
        ax.yaxis.set_label_text(
            "Survival Count",
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(
            duration_column,
            fontname="Arial",
            fontsize=10,
            color="#666666"
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
    
    
    # Create Cox Proportional hazard regression model
    model = CoxPHFitter()
    model.fit(
        df=dataframe, 
        duration_col=duration_column, 
        event_col="Survival Group"
    )
    model.print_summary()

    # Return the model
    return model

