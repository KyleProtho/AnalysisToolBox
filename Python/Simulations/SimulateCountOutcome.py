'''
Poisson distributions are discrete distributions that indicate the probability 
of a number of events

Conditions:
- Discrete non-negative data - count of events, the rate parameter can be a non-integer positive value
- Each event is independent of other events
- Each event happens at a fixed rate
- A fixed amount of time in which the events occur
'''

# Load packages
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

def SimulateCountOutcome(expected_count,
                         number_of_trials = 10000,
                         simulated_variable_name = 'Count',
                         random_seed = None,
                         plot_simulation_results = False):
    
    # Ensure arguments are valid
    if expected_count <= 0:
        raise ValueError("Please make sure that your expected_count_of_events argument is a positive whole number.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
        
    # Simulate count outcome
    df_simulation = pd.DataFrame(np.random.poisson(lam = expected_count,
                                                   size = number_of_trials),
                                 columns = [simulated_variable_name])
    
    # Generate plot if user requests it
    if plot_simulation_results == True:
        sns.set(style = "whitegrid")
        dcount = df_simulation[simulated_variable_name].nunique()
        if dcount >= 40:
            bin_setting = 40
        else:
            bin_setting = dcount
        sns.histplot(data = df_simulation,
                     x = simulated_variable_name, 
                     bins = bin_setting,
                     kde = True)
        plt.show()
    
    # Return simulation results 
    return df_simulation

# # Test
# df_test = SimulateCountOutcome(expected_count = 1)
# df_test = SimulateCountOutcome(expected_count = 5)
# df_test = SimulateCountOutcome(expected_count = 4,
#                                plot_simulation_results = True)
# df_test = SimulateCountOutcome(expected_count = 12,
#                                plot_simulation_results = True)
