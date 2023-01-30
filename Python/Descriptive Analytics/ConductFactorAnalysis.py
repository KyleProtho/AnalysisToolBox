# Load packages 
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
from logging import warning
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Declare function
def ConductFactorAnalysis(dataframe,
                          list_of_variables = None,
                          standardize_variables = True,
                          number_of_factors_to_create = None,
                          bartlett_sphericity_significance_level = 0.05,
                          measure_of_sampling_adequacy_threshold = 0.6,
                          pca_eigenvalue_threshold = 1):
    """_summary_
    This function conducts factor analysis on a dataset.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        list_of_variables (list, optional): _description_. The list of variables to be analyzed. Defaults to None, which means all variables will be analyzed.
        standardize_variables (bool, optional): _description_. Whether to standardize the variables prior to analysis. Defaults to True.
        number_of_factors_to_create (int, optional): _description_. The number of factors to create. Defaults to None, which means the number of factors will be determined automatically.
        bartlett_sphericity_significance_level (float, optional): _description_. The significance level for the Bartlett test for sphericity. Defaults to 0.05.
        measure_of_sampling_adequacy_threshold (float, optional): _description_. The threshold for the Keiser-Meyer-Olkin (KMO) test. Defaults to 0.6.
        pca_eigenvalue_threshold (int, optional): _description_. The threshold for the eigenvalue in the principal component analysis (PCA). Defaults to 1.
        
    Returns:
        Pandas dataframe: An updated Pandas dataframe with factor analysis results joined to the original dataset.
    """
    
    # Select list of variables to consider
    if list_of_variables == None:
        list_of_variables = dataframe.columns
        df_for_testing = dataframe.copy()
    else:
        df_for_testing = dataframe[list_of_variables].copy()
    
    # Filter to complete cases only
    df_for_testing = df_for_testing.dropna()
    
    # Run Bartlett test
    print("The Bartlett test for sphericity helps to determine whether there is a statistically detectable redundancy between the variables you selected. If so, those variables can be summarized with a few(er) number of factors.")
    bartlett_chi_square_value, bartless_p_value = calculate_bartlett_sphericity(df_for_testing)
    
    # See if Barlett test p-value is statistically significant. Show a warning if it is not.
    if bartless_p_value > bartlett_sphericity_significance_level:
        warning("The Bartlett test was not statistically significant at the " + str(bartlett_sphericity_significance_level) + " level -- meaning there is not a detectable redundancy among the variables you selected. Factor analysis is not recommended!")
    else:
        print("The Bartlett test was statistically significant at the " + str(bartlett_sphericity_significance_level) + " level -- meaning there is a detectable redundancy among the variables you selected.")
    
    # Run Keiser-Meyer-Olkin (KMO) test
    print("\n\nThe Keiser-Meyer-Olkin (KMO) test aims to estimate the proportion of variance among the variables you selected that may be explained by some underlying factor(s). The close the overall measure of sampling adequacy (MSA) is to 1, the more useful factor analysis will be in reducing your variables to a few factors.")
    kmo_all, kmo_model = calculate_kmo(df_for_testing)
    
    # See if KMO test result is below measure of sampling adequacy threshold
    if kmo_model < measure_of_sampling_adequacy_threshold:
        warning("The overall MSA is " + str(round(kmo_model, 3)) + ", which is below your desired threshold of " + str(measure_of_sampling_adequacy_threshold) + ". Factor analysis is not recommended!")
    else:
        print("The overall MSA is " + str(round(kmo_model, 3)) + ", which meets or exceeds your desired threshold of " + str(measure_of_sampling_adequacy_threshold) + ".")
    
    # Get list of variables that are above measure of sampling adequacy threshold
    df_variables = pd.DataFrame({
        'Variable': list_of_variables,
        'MSA': kmo_all
    })
    df_variables = df_variables[
        df_variables['MSA'] >= measure_of_sampling_adequacy_threshold
    ]
    list_of_eligible_variables = df_variables['Variable'].to_list()
    
    # Select variables that are above measure of sampling adequacy threshold
    df_for_fa = dataframe[list_of_eligible_variables].copy()
    
    # Filter to complete cases only
    df_for_fa = df_for_fa.dropna()
    
    # Standardize columns if requested
    if standardize_variables:
        df_transformed = StandardScaler().fit_transform(df_for_fa)
        df_transformed = pd.DataFrame(df_transformed)
        df_transformed.index = df_for_fa.index
        df_for_fa = df_transformed.copy()
        df_for_fa.columns = list_of_eligible_variables
    
    # Conduct principal component analysis
    fa = FactorAnalyzer()
    fa.fit(df_for_fa)
    
    # TO-DO: Show graph of variables
    
    # Get number of factors to use based on eigenvalue threshold (standard deviation)
    if number_of_factors_to_create == None:
        ev, v = fa.get_eigenvalues()
        number_of_factors_to_create = len(ev[ev >= pca_eigenvalue_threshold])
    
    # Conduct final factor analysis
    fa = FactorAnalyzer(
        n_factors=number_of_factors_to_create,
        rotation="verimax",
        method="minres",
        use_smc=True
    )
    fa.fit(df_for_fa)
    
    # Create list of factor names
    list_factor_names = []
    for i in range(0, number_of_factors_to_create):
        list_factor_names.append("Factor " + str(i + 1))
    
    # Show variance of each factor
    pd.DataFrame(
        fa.get_factor_variance(),
        columns=list_factor_names,
        index=["SS Loadings", "Share of Variance Explained (%)", "Cumulative Variance Explained"]
    )
    
    # Select variables that are above measure of sampling adequacy threshold
    df_factors = dataframe[list_of_eligible_variables].copy()
    
    # Filter to complete cases only
    df_factors = df_factors.dropna()
    
    # Predict factors for observed values in original dataset
    if standardize_variables:
        df_transformed = StandardScaler().fit_transform(df_factors)
        df_transformed = pd.DataFrame(df_transformed)
        df_transformed.index = df_factors.index
        df_factors = df_transformed.copy()
        df_factors.columns = list_of_eligible_variables
    df_predicted = fa.transform(df_factors)
    df_predicted = pd.DataFrame(
        df_predicted,
        columns=list_factor_names,
        index=df_factors.index
    )
    
    # Join predicted/fitted factors to original dataset
    dataframe = dataframe.merge(
        df_predicted,
        how='left',
        left_index=True,
        right_index=True
    )
    
    # Return dataset with factors added
    return(dataframe)
