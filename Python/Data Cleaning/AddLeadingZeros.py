import pandas as pd

def AddLeadingZeros(dataframe,
                    column_name,
                    fixed_length=None,
                    add_as_new_column=False):
    # If fixed length not specified, set the longest string as the fixed length
    if fixed_length == None:
        fixed_length = max(dataframe[column_name].astype(str).str.len())
    # If adding as new column, change the column name
    if add_as_new_column:
        new_column_name = column_name + ' - with leading 0s'
        dataframe[new_column_name] = dataframe[column_name].astype(str).str.zfill(fixed_length)
    else:
        dataframe[column_name] = dataframe[column_name].astype(str).str.zfill(fixed_length)
    # Return updated dataframe
    return(dataframe)