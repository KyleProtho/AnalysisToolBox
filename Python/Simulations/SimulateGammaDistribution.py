'''
Gamma distributions are continuous distributions that model the amount of time needed 
before a specified number of events happen.

Conditions:
- Continuous non-negative data
- A generalization of the exponential distribution, but more parameters to fit (With great flexibility, comes great complexity!)
- An exponential distribution models the time to the first event, the Gamma distribution models the time to the ‘n’th event.
'''

# Load packages
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

def SimulateGammaDistribution(shape=1,
                              expected_outcome=1,
                              number_of_trials=10000,
                              simulated_variable_name='Outcome',
                              random_seed=None,
                              plot_simulation_results=False):
    
    # Ensure arguments are valid
    if shape <= 0:
        raise ValueError("Please make sure that your number_of_events argument is greater than 0.")
    if expected_outcome <= 0:
        raise ValueError("Please make sure that your expected_time_between_events argument is greater than 0.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
        
    # Simulate time between events
    df_simulation = pd.DataFrame(np.random.gamma(shape=shape,
                                                 scale=expected_outcome,
                                                 size=number_of_trials),
                                 columns = [simulated_variable_name])
    
    # Generate plot if user requests it
    if plot_simulation_results == True:
        dcount = df_simulation[simulated_variable_name].nunique()
        if dcount >= 40:
            bin_setting = 40
        else:
            bin_setting = dcount
        sns.histplot(data=df_simulation,
                     x=simulated_variable_name, 
                     bins=bin_setting,
                     kde=True)
        plt.show()
    
    # Return simulation results 
    return df_simulation
