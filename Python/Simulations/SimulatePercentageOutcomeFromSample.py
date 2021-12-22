import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

def SimulatePercentageOutcomeFromSample(dataframe,
                                        outcome_variable,
                                        trials = 10000,
                                        plot_sim_results = True):
    # Remove nulls
    dataframe.columns = dataframe.columns.map(str)
    dataframe = dataframe[
        dataframe[outcome_variable].notnull()
    ]
    
    # Get parameters of beta distribution
    sim_params = stats.beta.fit(dataframe[outcome_variable])
    alpha_factor = sim_params[0]
    beta_factor = sim_params[1]
    print("Alpha factor used in simulation:", alpha_factor)
    print("Beta factor used in simulation:", beta_factor)
    
    # Conduct simulation
    arr_sim_results = np.random.beta(
        alpha_factor,
        beta_factor,
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
