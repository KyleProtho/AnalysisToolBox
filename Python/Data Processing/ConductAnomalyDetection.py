from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import textwrap

def ConductAnomalyDetection(dataframe, 
                            list_of_predictor_variables,
                            anomaly_threshold=0.95,
                            plot_detection_summary=True,
                            summary_plot_size=(20, 20),
                            column_name_for_anomaly_prob='Anomaly Probability',
                            column_name_for_anomaly_flag='Anomaly Detected'):
    """_summary_
    This function conducts anomaly detection on a dataset using the z-score method.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        list_of_predictor_variables (list): List of predictor variables to be analyzed.
        anomaly_threshold (float, optional): _description_. The threshold for the probability of an anomaly. Defaults to 0.95.
        plot_detection_summary (bool, optional): _description_. Whether to plot a summary of the anomaly detection. Defaults to True.
        summary_plot_size (tuple, optional): _description_. The size of the summary plot. Defaults to (20, 20).
        column_name_for_anomaly_prob (str, optional): _description_. The name of the column for the anomaly probability. Defaults to 'Anomaly Probability'.
        
    Returns:
        Pandas dataframe: Pandas dataframe containing the data to be analyzed with the anomaly probability and flag.
    """
    # If column_name_for_anomaly_prob is in the dataframe, drop it
    if column_name_for_anomaly_prob in dataframe.columns:
        dataframe = dataframe.drop(column_name_for_anomaly_prob, axis=1)
        
    # If column_name_for_anomaly_flag is in the dataframe, drop it
    if column_name_for_anomaly_flag in dataframe.columns:
        dataframe = dataframe.drop(column_name_for_anomaly_flag, axis=1)
    
    # Keep only the predictor variables
    dataframe_anomaly = dataframe[list_of_predictor_variables].copy()
    
    # Keep complete cases
    dataframe_anomaly = dataframe_anomaly.replace([np.inf, -np.inf], np.nan)
    dataframe_anomaly = dataframe_anomaly.dropna()
    
    # Get z-score from probability
    z_score = norm.ppf(anomaly_threshold / 2)
    
    # Iterate through each predictor variable
    dataframe_anomaly[column_name_for_anomaly_prob] = 1
    for predictor_variable in list_of_predictor_variables:
        # Get the mean and standard deviation of the predictor variable
        variable_mean = dataframe_anomaly[predictor_variable].mean()
        variable_std = dataframe_anomaly[predictor_variable].std()
        # Get z-score for each observation
        dataframe_anomaly[predictor_variable + ' z-score'] = (dataframe_anomaly[predictor_variable] - variable_mean) / variable_std
        # Update probability of anomaly
        dataframe_anomaly[column_name_for_anomaly_prob] *= norm.cdf(abs(dataframe_anomaly[predictor_variable + ' z-score']))
        # Drop the z-score column
        dataframe_anomaly = dataframe_anomaly.drop(predictor_variable + ' z-score', axis=1)
    
    # Join anomaly probability to original dataset
    dataframe = dataframe.merge(
        dataframe_anomaly[[column_name_for_anomaly_prob]], 
        left_index=True, 
        right_index=True
    )
    
    # Add flag for anomaly if probability is below threshold
    dataframe[column_name_for_anomaly_flag] = np.where(
        dataframe[column_name_for_anomaly_prob] > anomaly_threshold,
        True,
        False
    )
    
    # Show pairplot of the data
    if plot_detection_summary:
        # Generate a pairplot of the data
        plt.figure(figsize=summary_plot_size)
        sns.pairplot(
            data=dataframe[list_of_predictor_variables + [column_name_for_anomaly_flag]],
            hue=column_name_for_anomaly_flag
        )
        
        # Word wrap the axis labels
        for ax in plt.gcf().axes:
            ax.set_xlabel(textwrap.fill(ax.get_xlabel(), 40))
            ax.set_ylabel(textwrap.fill(ax.get_ylabel(), 40))
        
        # Show the plot
        plt.show()
    
    # Return the dataframe
    return(dataframe)


# # Test the function
# # dataset = ConductAnomalyDetection(
# #     dataframe=pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/5. Dimension Reduction/houses_to_rent.csv"),
# #     list_of_predictor_variables=['area', 'floor', 'parking spaces']
# # )
# dataset = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/5. Dimension Reduction/houses_to_rent.csv")
# # Mask some values in the dataset
# dataset = dataset.mask(np.random.random(dataset.shape) < .1)
# dataset = ConductAnomalyDetection(
#     dataframe=dataset,
#     list_of_predictor_variables=['area', 'floor', 'parking spaces']
# )
