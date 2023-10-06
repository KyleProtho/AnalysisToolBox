
# Load packages
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import binomial

def SimulateCountOfSuccesses(probability_of_success,
                             sample_size_per_trial = 1,
                             number_of_trials = 10000,
                             simulated_variable_name = 'Count of Successes',
                             random_seed = None,
                             plot_simulation_results = False):
    """
    The binomial distribution can be used to describe the number of successes 'p' in 'n' total events

    Conditions:
    - Discrete data
    - Two possible outcomes for each trial
    - Each trial is independent
    - The probability of success/failure is the same in each trial
    """
    
    # Ensure arguments are valid
    if probability_of_success >= 1:
        raise ValueError("Please change your probability_of_success argument -- it must be less than 1.")
    if sample_size_per_trial <= 0:
        raise ValueError("Please make sure that your sample_size_per_trial argument is a positive whole number.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)

    # Simulate T distributed outcome
    df_simulation = pd.DataFrame(binomial(n = sample_size_per_trial,
                                          p = probability_of_success,
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
# df_test = SimulateCountOfSuccesses(probability_of_success = 0.5)
# df_test = SimulateCountOfSuccesses(probability_of_success = 1.0)
# df_test = SimulateCountOfSuccesses(probability_of_success = 2.0)
# df_test = SimulateCountOfSuccesses(probability_of_success = 0.5,
#                                    sample_size_per_trial = 5)
# df_test = SimulateCountOfSuccesses(probability_of_success = 0.5,
#                                    sample_size_per_trial = 5,
#                                    plot_simulation_results = True)
# df_test = SimulateCountOfSuccesses(probability_of_success = 0.25,
#                                    sample_size_per_trial = 100,
#                                    plot_simulation_results = True)
