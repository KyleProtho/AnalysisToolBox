'''
A negative binomial distribution can be used to describe the number of successes ‘r - 1’ 
and ‘x’ failures in ‘x + r -1’ trials, until you have a success on the ‘x + r’th trial. 
Rephrased, this models the number of failures (x) you would have to see before you see a 
certain number of successes (r)

Conditions:
- Count of discrete events
- The events CAN be non-independent, implying that events can influence or cause other events
- Variance can exceed the mean
'''

# Load packages
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def SimulateCountUntilFirstSuccess(probability_of_success,
                                   number_of_trials = 10000,
                                   simulated_variable_name = 'Count Until First Success',
                                   random_seed = None,
                                   plot_simulation_results = False):
    
    # Ensure arguments are valid
    if probability_of_success > 1:
        raise ValueError("Please change your probability_of_success argument -- it must be less than or equal to 1.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)

    # Simulate count until first success
    list_sim_results = []
    for i in range(0, number_of_trials):
        is_success = False
        event_count = 0
        while is_success == False:
            event_count += 1
            sim_result = random.choices(population = [0, 1],
                                        weights = [1-probability_of_success, probability_of_success])
            sim_result = sum(sim_result)
            if sim_result == 1:
                list_sim_results.append(event_count)
                is_success = True
            else:
                continue
    df_simulation = pd.DataFrame(list_sim_results,
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
# df_test = SimulateCountUntilFirstSuccess(probability_of_success = 0.5,
#                                          number_of_trials = 100)
# df_test = SimulateCountUntilFirstSuccess(probability_of_success = 1.0)
# df_test = SimulateCountUntilFirstSuccess(probability_of_success = 2.0)
# df_test = SimulateCountUntilFirstSuccess(probability_of_success = 0.25)
# df_test = SimulateCountUntilFirstSuccess(probability_of_success = 0.5,
#                                          plot_simulation_results = True)
# df_test = SimulateCountUntilFirstSuccess(probability_of_success = 0.25,
#                                          plot_simulation_results = True)
