import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

def SimulateNormalDistributionFromSample(dataframe,
                                         outcome_variable,
                                         trials = 10000,
                                         plot_sim_results = True):
    # Remove nulls
    dataframe.columns = dataframe.columns.map(str)
    dataframe = dataframe[
        dataframe[outcome_variable].notnull()
    ]
    
    # Get parameters of beta distribution
    sim_param = stats.norm.fit(dataframe[outcome_variable])
    sim_param_mean = sim_param[0]
    sim_param_std = sim_param[1]
    print("Mean (expected value) of", outcome_variable, "used in simualtion:", sim_param_mean)
    print("Standard deviation of", outcome_variable, "used in simualtion:", sim_param_std)
    
    # Conduct simulation
    arr_sim_results = np.random.normal(
        sim_param_mean,
        sim_param_std,
        trials
    )
    df_sim_results = pd.DataFrame({
        outcome_variable: arr_sim_results
    })
    
    # If request, show plot
    if plot_sim_results:
        sns.histplot(
            df_sim_results[outcome_variable],
            kde=True,
            stat="density",
            label="Simulated"
        )
        sns.histplot(
            dataframe[outcome_variable],
            kde=True,
            stat="density",
            color="r",
            label="Sample"
        )
        plt.legend()
        plt.show()
    
    # Return simulation results
    return df_sim_results
