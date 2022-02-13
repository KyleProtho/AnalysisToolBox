import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def TestLogisticRegressionModelCI(model_dictionary,
                                  outcome_variable):
    # Join outcome and predictors of test datasets
    df_temp_test = model_dictionary["Outcome Test Dataset"].to_frame()
    df_temp_test = df_temp_test.merge(
        model_dictionary["Predictor Test Dataset"],
        how='left',
        left_index=True,
        right_index=True
    )
    
    # Add 95% CI estimates of model based on predictors (y = m*x + b = predictor*beta coeff + constant)
    for variable in model_dictionary["Predictor Test Dataset"].columns:
        lower_bound = model_dictionary["Fitted Model"].conf_int()[0][variable]
        upper_bound = model_dictionary["Fitted Model"].conf_int()[1][variable]
        best_guess = model_dictionary["Fitted Model"].params[variable]
        if variable == "const":
            df_temp_test['Lower bound'] = lower_bound
            df_temp_test['Upper bound'] = upper_bound
            df_temp_test['Best guess'] = best_guess
        else:
            df_temp_test['Lower bound'] = df_temp_test['Lower bound'] + (lower_bound * df_temp_test[variable])
            df_temp_test['Upper bound'] = df_temp_test['Upper bound'] + (upper_bound * df_temp_test[variable])
            df_temp_test['Best guess'] = df_temp_test['Best guess'] + (best_guess * df_temp_test[variable])
        df_temp_test['Lower bound'] = np.exp(df_temp_test['Lower bound']) / (1 + np.exp(df_temp_test['Lower bound']))
        df_temp_test['Upper bound'] = np.exp(df_temp_test['Upper bound']) / (1 + np.exp(df_temp_test['Upper bound']))
        df_temp_test['Best guess'] = np.exp(df_temp_test['Best guess']) / (1 + np.exp(df_temp_test['Best guess']))
    
    # Calculate different (residual) between best guess and observed outcome
    df_temp_test['Residual'] = df_temp_test[outcome_variable] - df_temp_test['Best guess']
    
    # Return tested dataset
    return df_temp_test


