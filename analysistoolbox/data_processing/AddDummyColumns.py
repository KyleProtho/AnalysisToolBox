# Load packages
import pandas as pd

# Declare function
def AddDummyColumns(dataframe,
                    categorical_column,
                    drop_first_group=False):
    """
    This function adds dummy columns to a dataset based on a categorical variable. The default is to add all dummy columns. If drop_first_group is set to True, the first dummy column is dropped.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe
        categorical_column (str): Name of column containing categorical variable
        drop_first_group (bool, optional): Whether the dummy variable of the first group should be dropped. Defaults to False.

    Returns:
        Pandas dataframe: An updated Pandas dataframe with dummy columns.
    """
    
    # Convert categorical variable to dummy variable
    df_temp = pd.get_dummies(dataframe[categorical_column],
                             prefix="Is",
                             drop_first=drop_first_group)
    
    # Column bind dummy variables to dataframe
    dataframe = pd.concat([dataframe.reset_index(drop=True), df_temp], axis=1)
    
    # Return dataframe
    return(dataframe)

