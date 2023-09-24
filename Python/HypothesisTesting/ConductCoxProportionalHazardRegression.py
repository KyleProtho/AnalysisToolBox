# Load packages
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter

# Declare function
def ConductCoxProportionalHazardRegression(dataframe,
                                           outcome_column,
                                           time_duration_column,
                                           list_of_predictors,
                                           plot_coefficients=True):
    # Select the columns to keep
    dataframe = dataframe[[outcome_column, time_duration_column] + list_of_predictors].copy()

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
    # Drop the survival group column
    dataframe = dataframe.drop(columns = [outcome_column])

    # Create Cox Proportional hazard regression model
    model = CoxPHFitter()
    model.fit(
        df=dataframe, 
        duration_col=time_duration_column, 
        event_col="Survival Group"
    )
    model.print_summary()

    # Plot the coefficients, if requested (TODO: Make plot prettier)
    if plot_coefficients:
        model.plot()

    # Return the model
    return model


# # Test function
# # cox_results = ConductCoxProportionalHazardRegression(
# #     dataframe = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/2. Cox Proportional Hazard Regression/lung.csv"),
# #     outcome_column="status",
# #     time_duration_column="time",
# #     list_of_predictors=[
# #         'age', 'sex', 'ph.ecog', 
# #         'ph.karno', 'pat.karno',
# #         'meal.cal', 'wt.loss'
# #     ]
# # )
# cox_results = ConductCoxProportionalHazardRegression(
#     dataframe = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/2. Cox Proportional Hazard Regression/lung.csv"),
#     outcome_column="status",
#     time_duration_column="time",
#     list_of_predictors=[
#         'sex', 'ph.ecog', 
#         'ph.karno', 'pat.karno',
#         'wt.loss'
#     ]
# )

