'''
The normal distribution is a continuous probability distribution that is symmetrical around its mean, 
most of the observations cluster around the central peak, and the probabilities for values further away 
from the mean taper off equally in both directions. Extreme values in both tails of the distribution 
are similarly unlikely.

Conditions:
- Continuous data
- Unbounded distribution
- Outliers are minimal
'''

# Load packages
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import normal

def SimulateNormallyDistributedOutcome(expected_outcome = 0,
                                       sd_of_outcome = 1,
                                       number_of_trials = 10000,
                                       simulated_variable_name = 'Simulated Outcome',
                                       random_seed = None,
                                       plot_simulation_results = False):
    
    # Ensure arguments are valid
    if sd_of_outcome < 0:
        raise ValueError("Please make sure that your sd_of_outcome argument is greater than or equal to 0.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
        
    # Simulate normally distributed outcome
    df_simulation = pd.DataFrame(normal(loc = expected_outcome,
                                        scale = sd_of_outcome,
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
# df_test = SimulateNormallyDistributedOutcome(expected_outcome = 0,
#                                              sd_of_outcome = 1)
# df_test = SimulateNormallyDistributedOutcome(expected_outcome = 0,
#                                              sd_of_outcome = -1)   # Should result in error
# df_test = SimulateNormallyDistributedOutcome(expected_outcome = 0,
#                                              sd_of_outcome = 1,
#                                              plot_simulation_results = True)
# df_test = SimulateNormallyDistributedOutcome(expected_outcome = 4,
#                                              sd_of_outcome = 1,
#                                              plot_simulation_results = True)
# df_test = SimulateNormallyDistributedOutcome(expected_outcome = 4,
#                                              sd_of_outcome = 10,
#                                              plot_simulation_results = True)
