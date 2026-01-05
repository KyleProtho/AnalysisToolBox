# Load packages
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Declare function
def ImputeMissingValuesUsingNearestNeighbors(dataframe,
                                             list_of_numeric_columns_to_impute,
                                             number_of_neighbors=3,
                                             averaging_method='uniform'):
    """
    Fill missing numeric data using K-Nearest Neighbors (KNN) imputation.

    This function uses scikit-learn's `KNNImputer` to estimate and fill missing values 
    based on the similarity of records across multiple dimensions. Each missing 
    value is replaced by the average (mean) value of the corresponding feature 
    from its 'k' nearest neighbors. This multivariate approach often provides more 
    accurate estimates than simple mean or median imputation, as it preserves 
    the relationships between different variables.

    The function is particularly useful for:
      * Preparing datasets for machine learning models that do not support missing values
      * Recovering missing information in multivariate datasets (e.g., sensor data)
      * Handling non-ignorable missing values where the missingness is correlated with other features
      * Scientific research where data loss occurs sporadically across samples
      * Financial modeling where specific indicators might be missing for certain periods
      * Pre-processing clinical data where laboratory results may be incomplete

    The function creates new columns with the suffix " - Imputed" to store the results, 
    preserving the original columns for auditability and comparison.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the numeric records to be imputed.
    list_of_numeric_columns_to_impute
        A list of column names containing numeric data with missing values. 
        Similarity is calculated based on the values across all columns in this list.
    number_of_neighbors
        The number of nearby samples to use for calculating the imputed value. 
        A larger 'k' provides a more global average, while a smaller 'k' is 
        more sensitive to local data patterns. Defaults to 3.
    averaging_method
        The weight function used in prediction. Options include:
          * 'uniform': All neighbors are weighted equally.
          * 'distance': Neighbors are weighted by the inverse of their distance; 
            closer neighbors have a greater influence.
        Defaults to 'uniform'.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional columns added for each variable in 
        `list_of_numeric_columns_to_impute`, labeled with the " - Imputed" suffix.

    Examples
    --------
    # Impute missing physiological data for patients
    import pandas as pd
    import numpy as np
    health_data = pd.DataFrame({
        'PatientID': [1, 2, 3, 4, 5],
        'Height_cm': [175, 160, 180, 165, 170],
        'Weight_kg': [80, 55, np.nan, 60, 72],
        'BMI': [26.1, 21.5, 27.2, np.nan, 24.9]
    })
    imputed_data = ImputeMissingValuesUsingNearestNeighbors(
        health_data, 
        ['Height_cm', 'Weight_kg', 'BMI'], 
        number_of_neighbors=2
    )
    # New columns 'Weight_kg - Imputed' and 'BMI - Imputed' are added.

    # Impute using distance weighting for more localized estimations
    sensor_readings = pd.DataFrame({
        'Temp': [20.5, 21.0, 19.8, np.nan, 20.2],
        'Humidity': [45, 48, np.nan, 46, 44],
        'Pressure': [1013, 1012, 1015, 1014, np.nan]
    })
    imputed_sensors = ImputeMissingValuesUsingNearestNeighbors(
        sensor_readings, 
        ['Temp', 'Humidity', 'Pressure'], 
        averaging_method='distance'
    )

    """
    
    # Select only the variables to impute
    dataframe_imputed = dataframe[list_of_numeric_columns_to_impute].copy()
    
    # Impute missing values using nearest neighbors
    imputer = KNNImputer(n_neighbors=number_of_neighbors,
                         weights=averaging_method)
    
    # Fit the imputer
    dataframe_imputed = imputer.fit_transform(dataframe_imputed)
    
    # Add "- Imputed" to the variable names
    list_new_column_names = []
    for variable in list_of_numeric_columns_to_impute:
        variable_imputed = variable + " - Imputed"
        list_new_column_names.append(variable_imputed)
    
    # Convert the imputed array to a dataframe
    dataframe_imputed = pd.DataFrame(dataframe_imputed,
                                     columns=list_new_column_names)
    
    # Bind the imputed dataframe to the original dataframe
    dataframe = pd.concat([dataframe, dataframe_imputed[list_new_column_names]], axis=1)
    
    # Return the dataframe with imputed values
    return(dataframe)

