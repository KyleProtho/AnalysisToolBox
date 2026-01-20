# Load packages
import pandas as pd
import numpy as np

# Declare function
def CreateSIPDataframe(name_of_items,
                       list_of_items,
                       number_of_trials=10000):
    """
    Generate a long-format DataFrame of stochastic trials (SIPs) for a list of items.

    This function creates a "Stochastic Information Packet" (SIP) structure, which is a fundamental
    component of Probabilistic Management. It generates a DataFrame where each item provided in 
    `list_of_items` is replicated for a specified number of trials. This provides a clean, 
    standardized foundation for conducting Monte Carlo simulations or probabilistic risk 
    assessments, ensuring that every simulated scenario (trial) is uniquely identifiable.

    Creating SIP DataFrames is essential for:
      * Healthcare: Simulating patient throughput scenarios across multiple hospital departments.
      * Intelligence Analysis: Modeling mission success probabilities across various regional sectors.
      * Cybersecurity: Simulating attack frequency and impact trials across different network nodes.
      * Supply Chain: Generating demand-risk scenarios for a list of distribution centers.
      * Epidemiology: Modeling infection spread trials across several demographic cohorts.
      * Project Management: Simulating completion times for a list of interdependent tasks.
      * Finance: Generating stochastic return profiles for a portfolio of different asset classes.
      * Quality Assurance: Simulating failure rates for multiple batches of manufactured components.

    Parameters
    ----------
    name_of_items : str
        The descriptive name for the column containing the items being simulated 
        (e.g., 'Hospital Ward', 'Asset Class', 'Region').
    list_of_items : list
        A list of unique identifiers or names for the objects being simulated.
    number_of_trials : int, optional
        The number of stochastic trials (simulated scenarios) to generate for each item. 
        Defaults to 10000.

    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with two columns: 'Trial' (the trial index from 1 to 
        `number_of_trials`) and the column specified by `name_of_items`.

    Examples
    --------
    # Healthcare: Creating a simulation shell for three different clinics
    clincs_sip = CreateSIPDataframe(
        name_of_items='Clinic Location',
        list_of_items=['North', 'South', 'East'],
        number_of_trials=1000
    )
    # This returns 3,000 rows (1,000 trials for each of the 3 clinics)

    # Intelligence: Simulating success/failure for various collection assets
    assets_sip = CreateSIPDataframe(
        name_of_items='Asset Type',
        list_of_items=['HUMINT', 'SIGINT', 'OSINT'],
        number_of_trials=5000
    )
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

