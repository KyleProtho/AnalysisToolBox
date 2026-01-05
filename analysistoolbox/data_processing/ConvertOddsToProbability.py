# Load packages
import numpy as np
import pandas as pd

# Declare function
def ConvertOddsToProbability(dataframe,
                             odds_column,
                             probability_column_name=None):
    """
    Convert odds to probability values in a pandas DataFrame.

    This function calculates the probability associated with a given set of odds using the
    standard statistical formula: p = odds / (1 + odds). Odds represent the ratio of the
    probability of an event occurring to the probability of it not occurring. Converting
    odds to probabilities is a common task in statistics, predictive modeling, and
    risk analysis.

    The function is particularly useful for:
      * Interpreting logistic regression output (converting odds ratios to probabilities)
      * Implied probability analysis in sports betting and financial markets
      * Risk assessment and epidemiological studies
      * Bayesian statistics and likelihood ratios
      * Data normalization for machine learning models
      * Communicating statistical risk to non-technical stakeholders

    The function handles missing values (NaN) gracefully and avoids division by zero
    errors if an odds value of -1 is encountered.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing a column with odds values.
    odds_column
        The name of the column in the DataFrame that contains the odds values.
        Odds should be numeric (int or float).
    probability_column_name
        The name for the new column that will contain the calculated probabilities.
        If None, the column will be named '{odds_column} - as probability'.
        Defaults to None.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column containing the calculated
        probabilities (ranging from 0 to 1). The original column is preserved.

    Examples
    --------
    # Convert simple betting odds to implied probabilities
    import pandas as pd
    import numpy as np
    betting_odds = pd.DataFrame({
        'outcome': ['Team A', 'Team B', 'Draw'],
        'odds': [1.5, 4.0, 2.0]
    })
    betting_odds = ConvertOddsToProbability(betting_odds, 'odds')
    # Adds 'odds - as probability' column: [0.6, 0.8, 0.666...]

    # Interpret logistic regression odds ratios
    model_results = pd.DataFrame({
        'feature': ['Age', 'Income', 'Education'],
        'odds_ratio': [1.05, 2.10, 1.45]
    })
    model_results = ConvertOddsToProbability(
        model_results, 
        'odds_ratio', 
        probability_column_name='probability'
    )

    # Handle missing values
    risks = pd.DataFrame({
        'risk_odds': [0.1, np.nan, 0.5, 9.0]
    })
    risks = ConvertOddsToProbability(risks, 'risk_odds')
    # Probability column: [0.09, NaN, 0.33, 0.9]

    """
    
    # If probability column name is not specified, set it to odds column name + "- as probability"
    if probability_column_name is None:
        probability_column_name = odds_column + " - as probability"
    
    # Convert odds to probability
    dataframe[probability_column_name] = np.where(
        (dataframe[odds_column].isnull()) | (dataframe[odds_column]+1 == 0),
        np.nan,
        dataframe[odds_column] / (1 + dataframe[odds_column])
    )
    
    # Return updated dataframe
    return dataframe

