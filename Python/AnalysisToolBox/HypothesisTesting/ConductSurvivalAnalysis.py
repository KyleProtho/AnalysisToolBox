# Load packages
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def ConductSurvivalAnalysis(dataframe,
                            outcome_column,
                            time_duration_column,
                            group_column=None,
                            return_time_table=True,
                            plot_survival_curve=True,
                            # Hypothesis testing arguments
                            conduct_log_rank_test=True,
                            significance_level=0.05,
                            print_log_rank_test_results=True,
                            # Line formatting arguments
                            line_color="#3269a8",
                            line_alpha=0.8,
                            sns_color_palette="Set2",
                            add_point_in_time_survival_curve=False,
                            point_in_time_survival_color="#3269a8",
                            # Text formatting arguments
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
                            # Plot formatting arguments
                            figure_size=(8, 6)):
    """This function conducts survival analysis on a dataframe.

    Args:
        dataframe (_type_): _description_
        outcome_column (_type_): _description_
        time_duration_column (_type_): _description_
        group_column (_type_, optional): _description_. Defaults to None.
        return_time_table (bool, optional): _description_. Defaults to True.
        plot_survival_curve (bool, optional): _description_. Defaults to True.
        conduct_log_rank_test (bool, optional): _description_. Defaults to True.
        significance_level (float, optional): _description_. Defaults to 0.05.
        print_log_rank_test_results (bool, optional): _description_. Defaults to True.
        line_color (str, optional): _description_. Defaults to "#3269a8".
        line_alpha (float, optional): _description_. Defaults to 0.8.
        sns_color_palette (str, optional): _description_. Defaults to "Set2".
        add_point_in_time_survival_curve (bool, optional): _description_. Defaults to True.
        point_in_time_survival_color (str, optional): _description_. Defaults to "#3269a8".
        title_for_plot (str, optional): _description_. Defaults to "Cumulative Survival Curve".
        subtitle_for_plot (_type_, optional): _description_. Defaults to None.
        caption_for_plot (_type_, optional): _description_. Defaults to None.
        data_source_for_plot (_type_, optional): _description_. Defaults to None.
        x_indent (float, optional): _description_. Defaults to -0.127.
        title_y_indent (float, optional): _description_. Defaults to 1.125.
        subtitle_y_indent (float, optional): _description_. Defaults to 1.05.
        caption_y_indent (float, optional): _description_. Defaults to -0.3.
        y_axis_label_indent (float, optional): _description_. Defaults to 0.78.
        x_axis_label_indent (float, optional): _description_. Defaults to 0.925.
        figure_size (tuple, optional): _description_. Defaults to (8, 6).

    Returns:
        _type_: _description_
    """
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


# Test the function
data = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/1. Survival Analysis/lung.csv")
# survival_analysis = ConductSurvivalAnalysis(
#     dataframe=data,
#     outcome_column="status",
#     time_duration_column="time"
# )
survival_analysis = ConductSurvivalAnalysis(
    dataframe=data,
    outcome_column="status",
    time_duration_column="time",
    group_column="sex"
)
# survival_analysis = ConductSurvivalAnalysis(
#     dataframe=data,
#     outcome_column="status",
#     time_duration_column="time",
#     group_column="ph.ecog"
# )
