# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import textwrap

# Declare function
def ConductSurvivalAnalysis(dataframe,
                            outcome_column,
                            time_duration_column,
                            group_column=None,
                            # Output arguments
                            return_time_table=True,
                            plot_survival_curve=True,
                            conduct_log_rank_test=True,
                            significance_level=0.05,
                            print_log_rank_test_results=True,
                            # Survival curve plot arguments
                            line_color="#3269a8",
                            line_alpha=0.8,
                            sns_color_palette="Set2",
                            add_point_in_time_survival_curve=False,
                            point_in_time_survival_color="#3269a8",
                            title_for_plot="Cumulative Survival Curve",
                            subtitle_for_plot='Shows the cumulative survival probability over time',
                            caption_for_plot=None,
                            data_source_for_plot=None,
                            x_indent=-0.127,
                            title_y_indent=1.125,
                            subtitle_y_indent=1.05,
                            caption_y_indent=-0.3,
                            y_axis_label_indent=0.78,
                            x_axis_label_indent=0.925,
                            figure_size=(8, 6)):
    """
    Conduct survival analysis to model time-to-event data and compare survival patterns.

    This function performs survival analysis using the Kaplan-Meier estimator to determine
    the probability of an event not occurring over a specified time duration. It provides
    detailed survival tables, optional log-rank tests for group comparisons, and high-quality
    visualizations of cumulative survival curves with confidence intervals.

    Survival analysis is essential for:
      * Analyzing customer churn and estimating subscription lifecycle
      * Evaluating patient outcomes and survival rates in clinical trials
      * Modeling time-to-failure in mechanical and engineering systems
      * Understanding employee retention patterns and attrition timing
      * Assessing credit risk and time-to-default for financial products
      * Tracking conversion cycles and time-to-purchase in sales funnels
      * Predicting project completion times and delivery durations

    The function fits Kaplan-Meier models using the `lifelines` library. For grouped
    data, it can perform both bivariate and multivariate log-rank tests to identify
    statistically significant differences in survival distributions between cohorts.
    The resulting visualization tracks cumulative survival probability, making it easy
    to identify when the risk of an event is highest across different segments.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the survival data to be analyzed.
    outcome_column
        Name of the column containing the binary event status (e.g., 1 for event
        occurred, 0 for censored).
    time_duration_column
        Name of the column containing the time-to-event or duration until the end of
        the observation period.
    group_column
        Optional name of the column used to segment the data into groups for
        comparative analysis. Defaults to None.
    return_time_table
        If True, includes the detailed event and survival probability table in the
        returned dictionary. Defaults to True.
    plot_survival_curve
        If True, generates and displays a Kaplan-Meier survival curve plot.
        Defaults to True.
    conduct_log_rank_test
        If True, performs a log-rank test to compare survival curves across groups
        when a `group_column` is provided. Defaults to True.
    significance_level
        The alpha level for statistical significance in the log-rank test.
        Defaults to 0.05.
    print_log_rank_test_results
        If True, prints a summary of the log-rank test results to the console.
        Defaults to True.
    line_color
        Color of the survival line for non-grouped analysis. Defaults to "#3269a8".
    line_alpha
        Transparency level for the survival lines and shaded confidence intervals,
        ranging from 0 to 1. Defaults to 0.8.
    sns_color_palette
        Seaborn color palette used for the lines and intervals when comparing
        groups. Defaults to "Set2".
    add_point_in_time_survival_curve
        If True, adds a secondary line to the plot showing point-in-time survival
        probabilities (non-grouped analysis only). Defaults to False.
    point_in_time_survival_color
        Color used for the point-in-time survival line. Defaults to "#3269a8".
    title_for_plot
        Main title text to display at the top of the plot. Defaults to
        "Cumulative Survival Curve".
    subtitle_for_plot
        Subtitle text to display below the main title. Defaults to
        'Shows the cumulative survival probability over time'.
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
    y_axis_label_indent
        Vertical position for the y-axis label. Defaults to 0.78.
    x_axis_label_indent
        Horizontal position for the x-axis label. Defaults to 0.925.
    figure_size
        Tuple specifying the (width, height) of the figure in inches. Defaults to (8, 6).

    Returns
    -------
    dict
        A dictionary containing the analysis results with the following keys:
          * 'survival_table': A pd.DataFrame containing event counts and survival
            probabilities over time.
          * [Group Names]: Individual `KaplanMeierFitter` objects for each group if
            `group_column` was provided.

    Examples
    --------
    # Simple survival analysis for machine failure
    import pandas as pd
    df = pd.DataFrame({
        'failed': [1, 0, 1, 1, 0] * 20,
        'hours': [100, 500, 250, 400, 600] * 20
    })
    results = ConductSurvivalAnalysis(
        dataframe=df,
        outcome_column='failed',
        time_duration_column='hours'
    )
    print(results['survival_table'].head())

    # Compare customer churn between subscription tiers
    subscription_df = pd.DataFrame({
        'churned': [1, 1, 0, 1, 0, 0] * 30,
        'months': [3, 6, 24, 1, 12, 18] * 30,
        'tier': ['Basic', 'Premium', 'Basic', 'Standard', 'Premium', 'Standard'] * 30
    })
    results = ConductSurvivalAnalysis(
        dataframe=subscription_df,
        outcome_column='churned',
        time_duration_column='months',
        group_column='tier',
        title_for_plot='Customer Retention by Subscription Tier',
        sns_color_palette='viridis'
    )

    # Statistical test for treatment efficacy without plotting
    results = ConductSurvivalAnalysis(
        dataframe=df,
        outcome_column='failed',
        time_duration_column='hours',
        plot_survival_curve=False,
        conduct_log_rank_test=True
    )

    """
    # Lazy load uncommon packages
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    
    # Select the columns to keep
    if group_column is None:
        dataframe = dataframe[[outcome_column, time_duration_column]].copy()
    else:
        dataframe = dataframe[[outcome_column, time_duration_column, group_column]].copy()

    # Remove records with missing values
    dataframe = dataframe.dropna()

    # Order the dataframe by outcome
    dataframe = dataframe.sort_values(by = [outcome_column, time_duration_column], ascending = [True, True])
    
    # Create a new column showing unqiue status as group starting with 0
    unique_status = dataframe[outcome_column].unique()
    dataframe["Survival Group"] = dataframe[outcome_column].apply(lambda x: list(unique_status).index(x))
    # Print the survival group and their corresponding status
    print("Outcome variables have been assigned accordingly to each group:")
    print(dataframe[["Survival Group", outcome_column]].drop_duplicates().reset_index(drop = True))

    # Calculate the survival probability (Kaplan-Meier Estimator)
    if group_column is None:
        model = KaplanMeierFitter()
        model.fit(
            durations=dataframe[time_duration_column], 
            event_observed=dataframe[outcome_column]
        )
    else:
        # Create dictionary to hold models for each group
        dict_models = {}
        for group in dataframe[group_column].unique():
            # Separate the dataframe by group
            group_data = dataframe[dataframe[group_column] == group].copy()
            # Fit the model for each group
            dict_models[group] = KaplanMeierFitter()
            dict_models[group].fit(
                durations=group_data[time_duration_column], 
                event_observed=group_data[outcome_column]
            )

    # Create a new dataframe to store the time table/survival probability
    if group_column is None:
        # Get event table from model
        data_time_table = model.event_table
        data_time_table = data_time_table.reset_index(drop=False)
        data_time_table.columns = data_time_table.columns.str.title()
        data_time_table.columns = data_time_table.columns.str.replace("_", " ")
        data_time_table = data_time_table.rename(columns={"Event At": time_duration_column})
        # Calculate the survival probability
        data_time_table["Survival Probability - Point in Time"] = (data_time_table["At Risk"] - data_time_table["Observed"]) / data_time_table["At Risk"]
        # Calculate the cumulative survival probability
        data_time_table["Survival Probability - Cumulative"] = model.survival_function_.values
        # Add confidence interval 
        data_time_table[["Survival Probability - Cumulative (Lower 95% C.I.)",  "Survival Probability - Cumulative (Upper 95% C.I.)"]] = model.confidence_interval_survival_function_.values
    else:
        # Sort the dataframe by group and time duration
        dataframe = dataframe.sort_values(by = [group_column, time_duration_column], ascending = [True, True])
        # Create dataframe to store the time table/survival probability
        data_time_table = pd.DataFrame()
        for group in dataframe[group_column].unique():
            group_model = dict_models[group]
            # Get event table from model
            group_data_time_table = group_model.event_table
            group_data_time_table = group_data_time_table.reset_index(drop=False)
            group_data_time_table.columns = group_data_time_table.columns.str.title()
            group_data_time_table.columns = group_data_time_table.columns.str.replace("_", " ")
            group_data_time_table = group_data_time_table.rename(columns={"Event At": time_duration_column})
            # Calculate the survival probability
            group_data_time_table["Survival Probability - Point in Time"] = (group_data_time_table["At Risk"] - group_data_time_table["Observed"]) / group_data_time_table["At Risk"]
            # Calculate the cumulative survival probability
            group_data_time_table["Survival Probability - Cumulative"] = group_model.survival_function_.values
            # Add confidence interval 
            group_data_time_table[["Survival Probability - Cumulative (Lower 95% C.I.)",  "Survival Probability - Cumulative (Upper 95% C.I.)"]] = group_model.confidence_interval_survival_function_.values
            # Add group column
            group_data_time_table[group_column] = group
            # Row bind to the dataframe
            data_time_table = pd.concat([data_time_table, group_data_time_table], axis=0).reset_index(drop=True)
            
    # If plot_survival_curve is True, plot the survival curve
    if plot_survival_curve:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Use Seaborn to create a line plot
        if group_column is None:
            sns.lineplot(
                data=data_time_table,
                x=time_duration_column,
                y="Survival Probability - Cumulative",
                color=line_color,
                alpha=line_alpha,
                ax=ax
            )
            # Add point in time curve, if requested
            if add_point_in_time_survival_curve:
                sns.lineplot(
                    data=data_time_table,
                    x=time_duration_column,
                    y="Survival Probability - Point in Time",
                    color=line_color,
                    alpha=line_alpha,
                    ax=ax
                )
            # Add confidence interval as a shaded area
            plt.fill_between(
                data_time_table[time_duration_column].values, 
                data_time_table["Survival Probability - Cumulative (Upper 95% C.I.)"].values,
                data_time_table["Survival Probability - Cumulative (Lower 95% C.I.)"].values,
                alpha=0.2,
                color=line_color
            )
        else:
            sns.lineplot(
                data=data_time_table,
                x=time_duration_column,
                y="Survival Probability - Cumulative",
                hue=group_column,
                palette=sns_color_palette,
                alpha=line_alpha,
                ax=ax
            )
            # Add confidence interval as a shaded area
            for i in range(data_time_table[group_column].unique().shape[0]):
                # Get current group
                group = data_time_table[group_column].unique().tolist()[i]
                # Get group color
                group_color = sns.color_palette(sns_color_palette)[i]
                # Filter to current group
                group_data_time_table = data_time_table[data_time_table[group_column] == group]
                # Add shaded area
                plt.fill_between(
                    group_data_time_table[time_duration_column].values, 
                    group_data_time_table["Survival Probability - Cumulative (Upper 95% C.I.)"].values,
                    group_data_time_table["Survival Probability - Cumulative (Lower 95% C.I.)"].values,
                    alpha=0.2,
                    color=group_color
                )
        
        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')

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
        ax.yaxis.set_label_coords(-0.1, y_axis_label_indent)
        ax.yaxis.set_label_text(
            "Survival Probability - Cumulative",
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(x_axis_label_indent, -0.1)
        ax.xaxis.set_label_text(
            time_duration_column,
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Add a word-wrapped caption if one is provided
        if caption_for_plot != None or data_source_for_plot != None:
            if caption_for_plot != None:
                # Word wrap the caption without splitting words
                if len(caption_for_plot) > 120:
                    # Split the caption into words
                    words = caption_for_plot.split(" ")
                    # Initialize the wrapped caption
                    wrapped_caption = ""
                    # Initialize the line length
                    line_length = 0
                    # Iterate through the words
                    for word in words:
                        # If the word is too long to fit on the current line, add a new line
                        if line_length + len(word) > 120:
                            wrapped_caption = wrapped_caption + "\n"
                            line_length = 0
                        # Add the word to the line
                        wrapped_caption = wrapped_caption + word + " "
                        # Update the line length
                        line_length = line_length + len(word) + 1
            else:
                wrapped_caption = ""
                
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

    # If there is a group column, run a log-rank test for each pair of groups
    if conduct_log_rank_test:
        if group_column != None:
            groups = dataframe[group_column].unique()
            # If there are more than two groups, run a multivariate log-rank test
            if len(groups) >= 3:
                results = multivariate_logrank_test(
                    dataframe[time_duration_column],
                    dataframe[group_column],
                    dataframe[outcome_column]
                )
                if print_log_rank_test_results:
                    results.print_summary()
            # If there are two groups, run a bivariate log-rank test
            elif len(groups) == 2:
                results = logrank_test(
                    dataframe[dataframe[group_column] == groups[0]][time_duration_column],
                    dataframe[dataframe[group_column] == groups[1]][time_duration_column],
                    dataframe[dataframe[group_column] == groups[0]][outcome_column],
                    dataframe[dataframe[group_column] == groups[1]][outcome_column]
                )
                if print_log_rank_test_results:
                    results.print_summary()
    
    # Create a dictionary to return models and survival tables
    return_dict = {}
    if group_column != None:
        for key, value in dict_models.items():
            return_dict[key] = value
    return_dict["survival_table"] = data_time_table
    
    # Return the dictionary
    return return_dict

