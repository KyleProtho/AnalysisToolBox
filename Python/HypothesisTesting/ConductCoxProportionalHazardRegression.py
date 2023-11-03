# Load packages
from lifelines import CoxPHFitter
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
def ConductCoxProportionalHazardRegression(dataframe,
                                           outcome_column,
                                           duration_column,
                                           list_of_predictor_columns,
                                           plot_survival_curve=True,
                                           # Survival curve plot formatting arguments
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


# # Test function
# data = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/2. Cox Proportional Hazard Regression/lung.csv")
# # cox_results = ConductCoxProportionalHazardRegression(
# #     dataframe = data,
# #     outcome_column="status",
# #     duration_column="time",
# #     list_of_predictor_columns=[
# #         'age', 'sex', 'ph.ecog', 
# #         'ph.karno', 'pat.karno',
# #         'meal.cal', 'wt.loss'
# #     ]
# # )
# cox_results = ConductCoxProportionalHazardRegression(
#     dataframe=data,
#     outcome_column="status",
#     duration_column="time",
#     list_of_predictor_columns=[
#         'sex', 'ph.ecog', 
#         'ph.karno', 'pat.karno',
#         'wt.loss'
#     ]
# )

