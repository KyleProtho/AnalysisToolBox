
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

def CreateBetaDistributionFromSample(dataframe,
                                     variable_name,
                                     plot_distribution=True,
                                     print_parameters=True,
                                     size_of_random_sample=10000):
    # Select the variable
    dataframe = dataframe[[variable_name]]
    
    # Filter out missing values
    dataframe = dataframe.dropna()
    
    # Fit a beta distribution to the data
    beta_distribution = stats.beta.fit(dataframe[variable_name])
    
    # Print the parameters of the beta distribution
    if print_parameters:
        print("Beta distribution parameters for " + variable_name + ":")
        print("a:", beta_distribution[0])
        print("b:", beta_distribution[1])
        print("loc:", beta_distribution[2])
        print("scale:", beta_distribution[3])
        
    
    # Plot the distribution of the variable and the fitted beta distribution using seaborn
    if plot_distribution:
        # Randomly sample from the beta distribution
        beta_sample = stats.beta.rvs(
            beta_distribution[0],
            beta_distribution[1],
            beta_distribution[2],
            beta_distribution[3],
            size=size_of_random_sample
        )
        beta_sample = pd.DataFrame(beta_sample, columns=[variable_name])
        
        # If necessary, duplicate the data to make the plot more readable
        while len(dataframe) < size_of_random_sample:
            if len(dataframe) * 2 < size_of_random_sample:
                dataframe = pd.concat([dataframe, dataframe]).reset_index(drop=True)
            else:
                # Randomly select a subset of the data
                random_addition = dataframe.sample(n=size_of_random_sample - len(dataframe), random_state=1)
                dataframe = pd.concat([dataframe, random_addition]).reset_index(drop=True)
        
        # Append the sample to the original dataframe
        dataframe['Source'] = 'Data'
        beta_sample['Source'] = 'Beta distribution'
        dataframe = pd.concat([dataframe, beta_sample]).reset_index(drop=True)
        
        # Create the plot
        sns.displot(
            data=dataframe, 
            x=variable_name,
            kind="kde", 
            hue="Source", 
            fill=True,
            legend=False
        )
        plt.legend(title=variable_name, loc='upper center' , labels=['Beta distribution', 'Data'])
        plt.title("Distribution of " + variable_name)
        plt.show()
        
    # Return the beta distribution parameters
    return beta_distribution


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# dist_petal_length = CreateBetaDistributionFromSample(iris, "petal length (cm)")    
