from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

def TestLinearRegressionModelCI(model_dictionary,
                                confidence_interval = .95,
                                show_plot = True):
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
        'mean': 'Predicted value',
        'mean_se': 'Standard error',
        'mean_ci_lower': ci_lower_column_name,
        'mean_ci_upper': ci_upper_column_name,
        'obs_ci_lower': pi_lower_column_name,
        'obs_ci_upper': pi_upper_column_name
    })
    
    # Add flag show if observation falls within 95% confidence interval
    outcome_variable = df_temp_test.columns[0]
    df_temp_test['Is within PI'] = np.where(
        (df_temp_test[outcome_variable] >= df_temp_test[pi_lower_column_name]) & (df_temp_test[outcome_variable] <= df_temp_test[pi_upper_column_name]),
        True,
        False
    )
    
    # Sort in ascending order of the Predicted mean
    df_temp_test = df_temp_test.sort_values(by=['Predicted value'])
    
    # Show plot of predictions if requested
    if show_plot:
        ax = sns.scatterplot(
            data=df_temp_test, 
            x='Predicted value',
            y=outcome_variable,
            hue='Is within PI',
            palette="Set2"
        )
        ax.fill_between(
            data=df_temp_test,
            x='Predicted value', 
            y1=pi_lower_column_name,
            y2=pi_upper_column_name,
            color="#58bf77", 
            alpha=0.3
        )
        ax.grid(False)
        ax.tick_params(axis='y', which='major', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylabel("Observed value")
        plt.suptitle(
            outcome_variable,
            x=0.125,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=14
        )
        plt.title(
            str(round(confidence_interval * 100, 0)) + "% prediction interval from regression model",
            loc='left',
            fontsize=10
        )
        # Show plot
        plt.show()
        # Clear plot
        plt.clf()
    
    # Return tested dataset
    return df_temp_test
