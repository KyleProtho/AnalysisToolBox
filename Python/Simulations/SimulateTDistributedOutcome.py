'''
The T distribution (also called Student's T Distribution) is a family of distributions that 
look almost identical to the normal distribution curve, only a bit shorter and fatter. 
A Student's t-distribution is used where one would be inclined to use a Normal distribution, 
but a Normal distribution is susceptible to outliers whereas a t-distribution is more robust.

Conditions:
- Continuous data
- Unbounded distribution
- Considered an overdispersed Normal distribution, mixture of individual normal distributions with different variances
'''

# Load packages
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import standard_t

def SimulateTDistributedOutcome(degrees_of_freedom,
                                expected_outcome = 0,
                                sd_of_outcome = 1,
                                number_of_trials = 10000,
                                simulated_variable_name = 'Simulated Outcome',
                                random_seed = None,
                                plot_simulation_results = False):
    
    # Ensure arguments are valid
    if sd_of_outcome < 0:
        raise ValueError("Please make sure that your sd_of_outcome argument is greater than or equal to 0.")
    if degrees_of_freedom <= 0:
        raise ValueError("Please make sure that your sd_of_outcome argument is greater than or equal to 1.")
    
    # If specified, set random seed for replicability
    if random_seed is not None:
        random.seed(random_seed)
        
    # Simulate T distributed outcome
    df_simulation = pd.DataFrame(standard_t(df = degrees_of_freedom,
                                            size = number_of_trials),
                                 columns = [simulated_variable_name])
    
    # Convert T score to original value scale
    df_simulation[simulated_variable_name] = df_simulation[simulated_variable_name] * sd_of_outcome + expected_outcome
    
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

# Test
# df_test = SimulateTDistributedOutcome(degrees_of_freedom = 1,
#                                       expected_outcome = 0,
#                                       sd_of_outcome = 1)
# df_test = SimulateTDistributedOutcome(degrees_of_freedom = 1,
#                                       expected_outcome = 0,
#                                       sd_of_outcome = -1)   # Should result in error
# df_test = SimulateTDistributedOutcome(degrees_of_freedom = 1,
#                                       expected_outcome = 0,
#                                       sd_of_outcome = 1,
#                                       plot_simulation_results = True)
# df_test = SimulateTDistributedOutcome(degrees_of_freedom = 1,
#                                       expected_outcome = 200,
#                                       sd_of_outcome = 50,
#                                       plot_simulation_results = True)
# df_test = SimulateTDistributedOutcome(degrees_of_freedom = 10,
#                                       expected_outcome = 200,
#                                       sd_of_outcome = 50,
#                                       plot_simulation_results = True)
