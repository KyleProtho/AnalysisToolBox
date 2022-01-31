import pandas as pd

def CountMissingData(dataframe):
    row_count = len(dataframe.index)
    for variable in dataframe.columns:
        null_row_count = dataframe[variable].isna().sum()
        null_row_share = round((null_row_count / row_count) * 100, 2)
        report_string = "Missing values in '" + variable + "': " + str(null_row_count) + " of " + str(row_count) + " rows (" + str(null_row_share) + "%)."
        print(report_string)
