import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def TestLinearRegressionModelCI(model_dictionary,
                                confidence_interval = .95):
    # Use model to generate prediction intervals
    model = model_dictionary["Fitted Model"]
    df_temp_test = model_dictionary["Predictor Test Dataset"]
    predictions = model.get_prediction(df_temp_test)
    predictions = predictions.summary_frame(alpha=1-confidence_interval)
    
    # Join predictions to observed outcomes of test datasets
    df_temp_test = model_dictionary["Outcome Test Dataset"].to_frame()
    df_temp_test = df_temp_test.merge(
        predictions,
        how='left',
        left_index=True,
        right_index=True
    )
    
    # Derive column names
    ci_lower_column_name = 'mean ' + str(round(confidence_interval * 100, 0)) + '% CI - lower bound'
    ci_upper_column_name = 'mean ' + str(round(confidence_interval * 100, 0)) + '% CI - upper bound'
    pi_lower_column_name = str(round(confidence_interval * 100, 0)) + '% PI - lower bound'
    pi_upper_column_name = str(round(confidence_interval * 100, 0)) + '% PI - upper bound'
    
    # Rename columns
    df_temp_test = df_temp_test.rename(columns={
        'mean_se': 'mean standard error',
        'mean_ci_lower': ci_lower_column_name,
        'mean_ci_upper': ci_upper_column_name,
        'obs_ci_lower': pi_lower_column_name,
        'obs_ci_upper': pi_upper_column_name
    })
    
    # Add flag show if observation falls within 95% confidence interval
    outcome_variable = df_temp_test.columns[0]
    df_temp_test['Is within PI'] = np.where(
        (df_temp_test[outcome_variable] >= df_temp_test[pi_lower_column_name]) & (df_temp_test[outcome_variable] <= df_temp_test[pi_upper_column_name]),
        1,
        0
    )
    
    # Return tested dataset
    return df_temp_test

