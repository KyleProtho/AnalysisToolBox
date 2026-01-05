# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import textwrap

# Declare function
def ConductAnomalyDetection(dataframe, 
                            list_of_columns_to_analyze,
                            anomaly_threshold=0.95,
                            plot_detection_summary=True,
                            summary_plot_size=(20, 20),
                            column_name_for_anomaly_prob='Anomaly Probability',
                            column_name_for_anomaly_flag='Anomaly Detected'):
    """
    Detect multivariate anomalies using z-score based probability analysis.

    This function performs statistical anomaly detection by calculating z-scores for multiple
    numeric variables and combining them to produce an overall anomaly probability for each
    row. The method assumes that normal data follows a Gaussian distribution and identifies
    observations that are statistically unlikely based on their distance from the mean across
    multiple dimensions. A binary flag is created to mark rows exceeding the anomaly threshold.

    The function is particularly useful for:
      * Fraud detection in financial transactions
      * Quality control in manufacturing processes
      * Network intrusion detection and cybersecurity monitoring
      * Sensor fault detection in IoT and industrial systems
      * Identifying outlier behavior in customer or user data
      * Medical diagnosis and health monitoring systems
      * Detecting unusual patterns in time series data

    The detection method calculates z-scores for each specified column, converts them to
    probabilities using the normal cumulative distribution function, and multiplies these
    probabilities to get a combined anomaly score. Lower probabilities indicate more unusual
    patterns. The optional pairplot visualization shows relationships between variables with
    anomalies highlighted in a different color.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the data to analyze. The DataFrame should include
        numeric columns specified in list_of_columns_to_analyze. Rows with missing values
        in these columns will be excluded from anomaly detection.
    list_of_columns_to_analyze
        List of column names (numeric variables) to use for anomaly detection. The function
        will calculate z-scores for each column and combine them to identify multivariate
        anomalies. All columns must contain numeric data.
    anomaly_threshold
        Probability threshold above which observations are flagged as anomalies. Values
        range from 0 to 1, where higher values are more sensitive (detect more anomalies).
        For example, 0.95 flags observations in the most extreme 5% of the distribution.
        Defaults to 0.95.
    plot_detection_summary
        If True, generates a seaborn pairplot visualization showing relationships between
        all analyzed variables, with anomalies color-coded. This helps visualize how
        anomalies differ from normal observations. Defaults to True.
    summary_plot_size
        Tuple specifying the (width, height) of the summary pairplot figure in inches.
        Only used if plot_detection_summary=True. Defaults to (20, 20).
    column_name_for_anomaly_prob
        Name for the new column that will contain the calculated anomaly probability
        (0 to 1, where values closer to 0 are more anomalous). If a column with this
        name already exists, it will be dropped. Defaults to 'Anomaly Probability'.
    column_name_for_anomaly_flag
        Name for the new binary column that flags detected anomalies (True/False).
        Rows with anomaly probability exceeding the threshold are marked as True.
        If a column with this name already exists, it will be dropped.
        Defaults to 'Anomaly Detected'.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with two additional columns:
          * {column_name_for_anomaly_prob}: Float values (0-1) indicating anomaly probability
          * {column_name_for_anomaly_flag}: Boolean values (True/False) indicating if the
            row is flagged as an anomaly based on the threshold
        Rows with missing values in analyzed columns will have NaN for these new columns.
        If plot_detection_summary=True, a pairplot is also displayed.

    Examples
    --------
    # Detect fraudulent transactions using multiple features
    import pandas as pd
    transactions = pd.DataFrame({
        'transaction_amount': [100, 150, 120, 5000, 130, 140, 110],
        'transaction_count_24h': [2, 3, 2, 15, 3, 2, 1],
        'avg_transaction_size': [50, 50, 60, 333, 43, 70, 110]
    })
    transactions = ConductAnomalyDetection(
        transactions,
        list_of_columns_to_analyze=['transaction_amount', 'transaction_count_24h', 'avg_transaction_size'],
        plot_detection_summary=False
    )
    # Likely flags row with $5000 transaction as anomalous

    # Monitor sensor readings with visualization
    sensor_data = pd.DataFrame({
        'temperature': [70, 72, 71, 73, 95, 72, 71],
        'pressure': [14.7, 14.8, 14.7, 14.9, 14.8, 14.7, 14.8],
        'vibration': [0.1, 0.12, 0.11, 0.13, 2.5, 0.12, 0.11]
    })
    sensor_data = ConductAnomalyDetection(
        sensor_data,
        list_of_columns_to_analyze=['temperature', 'pressure', 'vibration'],
        anomaly_threshold=0.99,  # More sensitive
        plot_detection_summary=True,
        summary_plot_size=(12, 12)
    )
    # Detects outlier with high temperature and vibration

    # Identify unusual customer behavior with custom column names
    customer_metrics = pd.DataFrame({
        'purchase_frequency': [2, 3, 2, 25, 3, 2],
        'avg_order_value': [50, 55, 48, 200, 52, 51],
        'return_rate': [0.05, 0.03, 0.04, 0.5, 0.04, 0.03]
    })
    customer_metrics = ConductAnomalyDetection(
        customer_metrics,
        list_of_columns_to_analyze=['purchase_frequency', 'avg_order_value', 'return_rate'],
        anomaly_threshold=0.95,
        plot_detection_summary=False,
        column_name_for_anomaly_prob='Anomaly Score',
        column_name_for_anomaly_flag='Is Suspicious'
    )
    # Uses custom column names for analysis results

    """
    
    # If column_name_for_anomaly_prob is in the dataframe, drop it
    if column_name_for_anomaly_prob in dataframe.columns:
        dataframe = dataframe.drop(column_name_for_anomaly_prob, axis=1)
        
    # If column_name_for_anomaly_flag is in the dataframe, drop it
    if column_name_for_anomaly_flag in dataframe.columns:
        dataframe = dataframe.drop(column_name_for_anomaly_flag, axis=1)
    
    # Keep only the predictor variables
    dataframe_anomaly = dataframe[list_of_columns_to_analyze].copy()
    
    # Keep complete cases
    dataframe_anomaly = dataframe_anomaly.replace([np.inf, -np.inf], np.nan)
    dataframe_anomaly = dataframe_anomaly.dropna()
    
    # Get z-score from probability
    z_score = norm.ppf(anomaly_threshold / 2)
    
    # Iterate through each predictor variable
    dataframe_anomaly[column_name_for_anomaly_prob] = 1
    for predictor_variable in list_of_columns_to_analyze:
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
        how='left', 
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
            data=dataframe[list_of_columns_to_analyze + [column_name_for_anomaly_flag]],
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

