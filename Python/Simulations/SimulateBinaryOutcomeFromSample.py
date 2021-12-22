import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

def SimulateBinaryOutcomeFromSample(dataframe,
                                    outcome_variable,
                                    trials = 10000,
                                    plot_sim_results = True):
    # Remove nulls
    dataframe.columns = dataframe.columns.map(str)
    dataframe = dataframe[
        dataframe[outcome_variable].notnull()
    ]
    
    # Get parameters of Bernoulli distribution
    sim_param = dataframe[outcome_variable].mean()
    print("Probability of success (coded as 1) used in simualtion:", sim_param)
    
    # Conduct simulation
    arr_sim_results = np.random.binomial(
        1,
        sim_param,
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
            label="Simulated",
            bins=2
        )
        sns.histplot(
            dataframe[outcome_variable],
            kde=True,
            stat="density",
            color="r",
            label="Sample",
            bins=2
        )
        plt.legend()
        plt.show()
    
    # Return simulation results
    return df_sim_results
