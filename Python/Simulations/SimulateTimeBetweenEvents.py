'''
Exponential distributions are continuous distributions that model the duration of events

Conditions:
- Continuous non-negative data
- Time between events are considered to happen at a constant rate
- Events are considered to be independent
'''

# Load packages
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import exponential

def SimulateTimeBetweenEvents(expected_time_between_events = 1,
                              number_of_trials = 10000,
                              simulated_variable_name = 'Time Between Events',
                              random_seed = None,
                              plot_simulation_results = False):
    
    # Ensure arguments are valid
    if expected_time_between_events <= 0:
        raise ValueError("Please make sure that your expected_time_between_events argument is greater than 0.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
        
    # Simulate time between events
    df_simulation = pd.DataFrame(exponential(scale = expected_time_between_events,
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
# df_test = SimulateTimeBetweenEvents(expected_time_between_events = 1)
# df_test = SimulateTimeBetweenEvents(expected_time_between_events = -1)  ## Should result in error
# df_test = SimulateTimeBetweenEvents(expected_time_between_events = 1,
#                                     plot_simulation_results = True)
# df_test = SimulateTimeBetweenEvents(expected_time_between_events = 0.5,
#                                     plot_simulation_results = True)
# df_test = SimulateTimeBetweenEvents(expected_time_between_events = 2,
#                                     plot_simulation_results = True)
# df_test = SimulateTimeBetweenEvents(expected_time_between_events = 20,
#                                     plot_simulation_results = True)
