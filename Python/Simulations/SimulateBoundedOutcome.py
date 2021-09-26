'''
The beta distribution is commonly used in project planning simulations.  It has an 
absolute upper and lower bound, but, like the triangular distribution, has a mode 
that results are much more likely to be near.  The mode can be anywhere between the 
minimum and maximum value. 

Conditions:
- Possible outcomes are bounded.
'''

# Load packages
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linspace
from numpy.random import beta

def SimulateBoundedOutcome(expected_outcome,
                           minimum_outcome,
                           maximum_outcome,
                           number_of_trials = 10000,
                           simulated_variable_name = 'Simulated Outcome',
                           random_seed = None,
                           plot_simulation_results = False):
    
    # Ensure arguments are valid
    if expected_outcome <= minimum_outcome:
        raise ValueError("Please make sure that your expected_outcome argument is greater than your minimum_outcome argument.")
    if expected_outcome >= maximum_outcome:
        raise ValueError("Please make sure that your expected_outcome argument is less than your maximum_outcome argument.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
    
    # The following parameters are calculated based on formulas from the Beta Sim in the Excel file from How to Measure Anything
    ## Calculate relative mean
    relative_mean = ((expected_outcome - minimum_outcome) / (maximum_outcome - minimum_outcome) * 4 + 1) / 6
    ## Calculate alpha parameter
    alpha_param = relative_mean ** 2 *(1 - relative_mean) * 6 ** 2 - 1
    if alpha_param <= 0:
        alpha_param = 0.001
    ## Calculate beta parameter
    beta_param = (1 - relative_mean) / relative_mean * alpha_param
    
    # Simulate bounded outcome
    df_simulation = pd.DataFrame(beta(a = alpha_param, 
                                      b = beta_param,
                                      size = number_of_trials),
                                 columns = [simulated_variable_name])
    df_simulation[simulated_variable_name] = round(df_simulation[simulated_variable_name], 3)
    
    # Rescale simulated outcomes to original values
    df_value_lookup = pd.DataFrame({
        'Original Value': linspace(start = minimum_outcome, stop = maximum_outcome, num = 1000),
        simulated_variable_name: linspace(start = 0, stop = 1,  num = 1000)
    })
    df_value_lookup[simulated_variable_name] = round(df_value_lookup[simulated_variable_name], 3)
    df_simulation = df_simulation.merge(df_value_lookup, 
                                        on = simulated_variable_name, 
                                        how = 'left')
    
    # Drop original simulated value and rename joined value
    df_simulation = df_simulation.drop([simulated_variable_name], axis=1)
    df_simulation = df_simulation.rename(columns = {'Original Value': simulated_variable_name})
    
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
# df_test = SimulateBoundedOutcome(expected_outcome = 0.5,
#                                  minimum_outcome = 0,
#                                  maximum_outcome = 1)
# df_test = SimulateBoundedOutcome(expected_outcome = 0,
#                                  minimum_outcome = 0,
#                                  maximum_outcome = 1)   # Should result in error
# df_test = SimulateBoundedOutcome(expected_outcome = 0.5,
#                                  minimum_outcome = 0,
#                                  maximum_outcome = 1,
#                                  plot_simulation_results = True)
# df_test = SimulateBoundedOutcome(expected_outcome = 16,
#                                  minimum_outcome = 10,
#                                  maximum_outcome = 28,
#                                  plot_simulation_results = True)
