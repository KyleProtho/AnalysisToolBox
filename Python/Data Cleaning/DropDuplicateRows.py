import pandas as pd

def DropDuplicateRows(dataframe):
    # Show row count before and after dropping duplicates
    print("Number of rows before dropping duplicates: " + str(len(dataframe.index)))
    dataframe = dataframe.drop_duplicates()
    print("Number of rows after dropping duplicates: " + str(len(dataframe.index)))
    
    # Reutrn updated dataframe
    return(dataframe)