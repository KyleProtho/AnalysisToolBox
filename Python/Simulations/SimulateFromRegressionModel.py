from random import random
import pandas as pd
import statsmodels.api as sm
# Must import SimulateNormallyDistributedOutcome function

def SimulateFromRegressionModel(model_dictionary,
                                outcome_variable,
                                list_of_predictors,
                                list_of_predictor_values,
                                number_of_trials = 10000,
                                random_seed = None):
    # Create empty dataframe
    df_temp = pd.DataFrame()
    
    # Iterate through list of predictors
    for i in range(len(list_of_predictors)):
        # Create beta coef. column name for predictor
        beta_coef_colname = 'Beta Coef - ' + list_of_predictors[i]
        # Generate distribution of predictor's beta coef.
        if random_seed != None:
            if i == 0:
                df_temp = SimulateNormallyDistributedOutcome(
                    expected_outcome = model_dictionary['Fitted Model'].params[list_of_predictors[i]],
                    sd_of_outcome = model_dictionary['Fitted Model'].bse[list_of_predictors[i]],
                    number_of_trials = number_of_trials,
                    simulated_variable_name = beta_coef_colname,
                    random_seed = random_seed
                )
            else:
                df_temp = df_temp.join(
                    SimulateNormallyDistributedOutcome(
                        expected_outcome = model_dictionary['Fitted Model'].params[list_of_predictors[i]],
                        sd_of_outcome = model_dictionary['Fitted Model'].bse[list_of_predictors[i]],
                        number_of_trials = number_of_trials,
                        simulated_variable_name = beta_coef_colname,
                        random_seed = random_seed
                    )
                )
        else:
            df_temp = SimulateNormallyDistributedOutcome(
                    expected_outcome = model_dictionary['Fitted Model'].params[list_of_predictors[i]],
                    sd_of_outcome = model_dictionary['Fitted Model'].bse[list_of_predictors[i]],
                    number_of_trials = number_of_trials,
                    simulated_variable_name = beta_coef_colname
                )
        # Add predictor vriable value
        df_temp[list_of_predictors[i]] = list_of_predictor_values[i]
    
    # Generate distribution of y-intercept/constant
    if random_seed != None:
        df_temp = df_temp.join(
            SimulateNormallyDistributedOutcome(
                expected_outcome = model_dictionary['Fitted Model'].params['const'],
                sd_of_outcome = model_dictionary['Fitted Model'].bse['const'],
                number_of_trials = number_of_trials,
                simulated_variable_name = 'const',
                random_seed = random_seed
            )
        )
    else:
        df_temp = df_temp.join(
            SimulateNormallyDistributedOutcome(
                expected_outcome = model_dictionary['Fitted Model'].params['const'],
                sd_of_outcome = model_dictionary['Fitted Model'].bse['const'],
                number_of_trials = number_of_trials,
                simulated_variable_name = 'const'
            )
        )
    
    # Generate estimate for outcome variable
    df_temp[outcome_variable] = df_temp['const']
    for i in range(len(list_of_predictors)):
        # Create beta coef. column name for predictor
        beta_coef_colname = 'Beta Coef - ' + list_of_predictors[i]
        # Add regression terms to estimate
        df_temp[outcome_variable] = df_temp[outcome_variable] + (df_temp[list_of_predictors[i]] * df_temp[beta_coef_colname])
        
    # Return results
    return(df_temp)