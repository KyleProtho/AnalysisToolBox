# Load packages
import pandas as pd
import numpy as np

# Declare function
def CreateSIPDataframe(name_of_items,
                       list_of_items,
                       number_of_trials=10000):
    """
    Creates a pandas DataFrame of trials of simulations (i.e., stochastic information packet, or SIP) for each item specified in a list.

    Args:
        name_of_items (str): The name of the item being simulated.
        list_of_items (list): A list of items to simulate.
        number_of_trials (int, optional): The number of trials to simulate. Defaults to 10000.

    Returns:
        pandas.DataFrame: A DataFrame containing the simulation results.
    """
    
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
        df_sips = pd.concat([df_sips, df_temp])
        
    # Return SIP DataFrame
    return(df_sips)

