# Load packages
import pandas as pd

# Declare function
def AddDummyColumns(dataframe,
                    categorical_variable_column_name,
                    drop_first_group=True):
    # Convert categorical variable to dummy variable
    df_temp = pd.get_dummies(dataframe[categorical_variable_column_name],
                             prefix="Is",
                             drop_first=drop_first_group)
    
    # Column bind dummy variables to dataframe
    dataframe = pd.concat([dataframe.reset_index(drop=True), df_temp], axis=1)
    
    # Return dataframe
    return(dataframe)
