# Load packages
import pandas as pd
import numpy as np

def CreateSIPDataframe(name_of_items,
                       list_of_items,
                       number_of_trials=10000):
    # Create dataframe of trials
    df_sips = pd.DataFrame(columns=[
        'Trial',
        name_of_items
    ])
    
    # Iterate through list of items
    for item in list(set(list_of_items)):
        # Create trial column with item name
        df_temp = pd.DataFrame({
            'Trial': np.linspace(1, number_of_trials, number_of_trials, dtype='int'),
            name_of_items: item
        })
        # Append to SIP dataframe
        df_sips = df_sips.append(df_temp)
        
    # Return SIP DataFrame
    return(df_sips)