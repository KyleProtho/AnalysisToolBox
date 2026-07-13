# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Declare function
def ConductLinearRegressionAnalysis(dataframe,
                                    outcome_variable,
                                    list_of_predictors,
                                    add_constant=False,
                                    scale_predictors=False,
                                    variable_selection='none',
                                    selection_limit=4,
                                    selection_p_threshold=0.05,
                                    show_diagnostic_plots_for_each_predictor=False,
                                    show_help=True):
    """
    Conduct linear regression analysis to model relationships between variables and assess significance.

    This function performs Ordinary Least Squares (OLS) linear regression to examine how
    one or more predictor variables influence a continuous outcome variable. It provides
    statistical summaries, regression coefficients, and diagnostic visualizations to
    help interpret the strength and nature of the relationships in the data.

    Linear regression analysis is essential for:
      * Estimating the impact of independent variables on a continuous outcome
      * Predictive modeling for numerical values and trend forecasting
      * Testing scientific and economic hypotheses regarding variable associations
      * Analyzing the drivers of business metrics like revenue or customer satisfaction
      * Social science research on the effects of demographic or environmental factors
      * Identifying key features and their relative importance in a dataset
      * Validating the assumptions of linear relationships between variables

    The function automatically handles preprocessing by removing missing (NaN) and
    infinite values. It supports optional feature scaling using `StandardScaler`
    and can generate detailed diagnostic plots for each predictor variable using
    `statsmodels` diagnostics.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the variables for the regression analysis.
    outcome_variable
        Name of the column in the DataFrame representing the dependent (outcome) variable.
    list_of_predictors
        A list of column names for the independent (predictor) variables to include in
        the model.
    add_constant
        If True, adds an intercept (constant term) to the model. Defaults to False.
    scale_predictors
        If True, scales the predictor variables to have a mean of 0 and standard
        deviation of 1 using `StandardScaler` before fitting the model. Defaults to False.
    variable_selection
        Stepwise variable selection method to apply before fitting the final model.
        Options:
          * 'none'     — No selection; use all predictors in ``list_of_predictors``.
          * 'forward'  — Start from the null model (intercept only) and greedily add
            the predictor that most lowers AIC, stopping when no addition helps or
            ``selection_limit`` is reached.
          * 'backward' — Start with all predictors and iteratively remove the one with
            the highest p-value until all remaining p-values fall below
            ``selection_p_threshold``.
          * 'mixed'    — Forward steps add variables by AIC improvement; after each
            addition, any in-model predictor whose p-value exceeds
            ``selection_p_threshold`` is removed. Continues until stable.
        Defaults to 'none'.
    selection_limit
        Hard cap on the number of predictors kept by 'forward' or 'mixed' selection.
        Defaults to 4.
    selection_p_threshold
        P-value threshold used by 'backward' and 'mixed' selection. Predictors with
        p-value at or above this threshold are candidates for removal. Defaults to 0.05.
    show_diagnostic_plots_for_each_predictor
        If True, displays diagnostic regression plots (e.g., partial regression plots)
        for each predictor in the model. Defaults to False.
    show_help
        If True, prints a quick guide to the console explaining how to access and
        interpret the output dictionary. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing the regression results with the following keys:
          * 'Fitted Model': The results object from the statsmodels OLS fit.
          * 'Model Summary': A comprehensive statistical summary of the regression model.

    Examples
    --------
    # Analyze the impact of experience and education on salary
    import pandas as pd
    df = pd.DataFrame({
        'salary': [50000, 60000, 55000, 80000, 75000] * 10,
        'experience_years': [1, 5, 3, 10, 8] * 10,
        'education_level': [12, 16, 14, 18, 16] * 10
    })
    results = ConductLinearRegressionAnalysis(
        dataframe=df,
        outcome_variable='salary',
        list_of_predictors=['experience_years', 'education_level'],
        add_constant=True
    )
    print(results['Model Summary'])

    # Perform regression with feature scaling and diagnostic plots
    advertising_df = pd.DataFrame({
        'sales': [22, 10, 9, 18, 12] * 10,
        'tv_spend': [230, 44, 17, 151, 180] * 10,
        'radio_spend': [37, 39, 45, 41, 10] * 10
    })
    results = ConductLinearRegressionAnalysis(
        dataframe=advertising_df,
        outcome_variable='sales',
        list_of_predictors=['tv_spend', 'radio_spend'],
        add_constant=True,
        scale_predictors=True,
        show_diagnostic_plots_for_each_predictor=True
    )

    # Simple bivariate regression without intercept
    results = ConductLinearRegressionAnalysis(
        dataframe=df,
        outcome_variable='salary',
        list_of_predictors=['experience_years'],
        add_constant=False
    )

    """
    
    # Select columns specified
    dataframe = dataframe[list_of_predictors + [outcome_variable]]
    
    # Remove NAN and inf values
    dataframe.dropna(inplace=True)
    dataframe = dataframe[np.isfinite(dataframe).all(1)]

    # Variable selection (forward / backward / mixed)
    if variable_selection in ('forward', 'backward', 'mixed'):
        print("\n" + "=" * 62)
        print(f"  {variable_selection.upper()} SELECTION")
        print("=" * 62)
        print(
            "\n  Warning: Forward selection is a greedy approach, and might include\n"
            "  variables early that later become redundant.\n"
        )

        if variable_selection in ('forward', 'mixed'):
            print("  What is AIC?  (Akaike Information Criterion)")
            print("  " + "-" * 46)
            print(
                "  AIC measures how well a model fits the data while penalizing\n"
                "  complexity. Every predictor you add reduces residual error\n"
                "  (RSS), but AIC charges a 'complexity penalty' of 2 points per\n"
                "  new parameter. A predictor earns its place only if its error\n"
                "  reduction outweighs that cost.\n"
                "  Lower AIC = better model. A difference of 2 points is\n"
                "  noticeable; 10+ is substantial."
            )
            if variable_selection == 'mixed':
                print()

        if variable_selection in ('backward', 'mixed'):
            print("  P-value threshold: {:.2f}".format(selection_p_threshold))
            print("  " + "-" * 46)
            print(
                "  A predictor's p-value is the probability of observing its\n"
                "  estimated coefficient -- or one more extreme -- purely by\n"
                "  chance, assuming the predictor has no true effect.\n"
                "  p < {t:.2f}  ->  statistically significant; keep the predictor.\n"
                "  p >= {t:.2f} ->  likely noise; remove the predictor.".format(
                    t=selection_p_threshold
                )
            )

        # Null model (intercept only) used as AIC baseline for forward/mixed
        null_X = pd.DataFrame({"const": np.ones(len(dataframe))}, index=dataframe.index)
        null_aic = sm.OLS(dataframe[outcome_variable], null_X).fit().aic

        # ── FORWARD SELECTION ────────────────────────────────────────
        if variable_selection == 'forward':
            remaining = list_of_predictors.copy()
            selected = []
            current_aic = null_aic
            print(f"\n  Null model AIC (intercept only): {current_aic:.4f}")
            print(f"  Candidates:  {remaining}")
            print(f"  Limit:       {selection_limit}\n")

            while remaining and len(selected) < selection_limit:
                best_aic, best_var = current_aic, None
                for var in remaining:
                    try:
                        cand_aic = sm.OLS(
                            dataframe[outcome_variable],
                            sm.add_constant(dataframe[selected + [var]])
                        ).fit().aic
                    except Exception:
                        continue
                    if cand_aic < best_aic:
                        best_aic, best_var = cand_aic, var
                if best_var is None:
                    print("  Stopped: no remaining predictor improves AIC.")
                    break
                selected.append(best_var)
                remaining.remove(best_var)
                print(f"  Step {len(selected)}: Added '{best_var}' | AIC: {best_aic:.4f} | Improved by {current_aic - best_aic:.4f}")
                current_aic = best_aic

            if len(selected) == selection_limit and remaining:
                print(f"\n  Stopped: reached the limit of {selection_limit} predictor(s).")
            list_of_predictors = selected

        # ── BACKWARD SELECTION ───────────────────────────────────────
        elif variable_selection == 'backward':
            current_preds = list_of_predictors.copy()
            print(f"\n  Starting predictors: {current_preds}")
            print(f"  P-value threshold:   {selection_p_threshold}\n")
            step = 0

            while len(current_preds) > 0:
                X_curr = sm.add_constant(dataframe[current_preds])
                pvals = sm.OLS(dataframe[outcome_variable], X_curr).fit().pvalues[current_preds]
                worst_var = pvals.idxmax()
                worst_p = pvals.max()
                if worst_p <= selection_p_threshold:
                    print(f"  All remaining predictors have p-value <= {selection_p_threshold}. Stopping.")
                    break
                step += 1
                current_preds.remove(worst_var)
                label = current_preds if current_preds else ['(none)']
                print(f"  Step {step}: Removed '{worst_var}' | p-value: {worst_p:.4f} | Remaining: {label}")

            list_of_predictors = current_preds

        # ── MIXED SELECTION ──────────────────────────────────────────
        elif variable_selection == 'mixed':
            remaining = list_of_predictors.copy()
            selected = []
            current_aic = null_aic
            print(f"\n  Null model AIC (intercept only): {current_aic:.4f}")
            print(f"  Candidates: {remaining}")
            print(f"  Limit: {selection_limit}  |  P-value threshold: {selection_p_threshold}\n")

            step = 0
            seen_states = set()
            max_iter = (len(list_of_predictors) + 1) * 4

            for _ in range(max_iter):
                state = frozenset(selected)
                if state in seen_states:
                    print("  Converged: model has stabilized.")
                    break
                seen_states.add(state)
                forward_taken = False
                backward_taken = False

                # Forward step: add best predictor by AIC
                if remaining and len(selected) < selection_limit:
                    best_aic, best_var = current_aic, None
                    for var in remaining:
                        try:
                            cand_aic = sm.OLS(
                                dataframe[outcome_variable],
                                sm.add_constant(dataframe[selected + [var]])
                            ).fit().aic
                        except Exception:
                            continue
                        if cand_aic < best_aic:
                            best_aic, best_var = cand_aic, var
                    if best_var is not None:
                        selected.append(best_var)
                        remaining.remove(best_var)
                        step += 1
                        print(f"  Step {step} [+]: Added '{best_var}' | AIC: {best_aic:.4f} | Improved by {current_aic - best_aic:.4f}")
                        current_aic = best_aic
                        forward_taken = True

                # Backward step: remove worst predictor if p-value exceeds threshold
                if len(selected) >= 1:
                    try:
                        X_curr = sm.add_constant(dataframe[selected])
                        pvals = sm.OLS(dataframe[outcome_variable], X_curr).fit().pvalues[selected]
                        worst_var = pvals.idxmax()
                        worst_p = pvals.max()
                        if worst_p > selection_p_threshold:
                            selected.remove(worst_var)
                            remaining.append(worst_var)
                            step += 1
                            print(f"  Step {step} [-]: Removed '{worst_var}' | p-value: {worst_p:.4f}")
                            current_aic = (
                                sm.OLS(dataframe[outcome_variable],
                                       sm.add_constant(dataframe[selected])).fit().aic
                                if selected else null_aic
                            )
                            backward_taken = True
                    except Exception:
                        pass

                if not forward_taken and not backward_taken:
                    print("  Converged: no further additions or removals improve the model.")
                    break

            if len(selected) == selection_limit and remaining:
                print(f"\n  Stopped: reached the limit of {selection_limit} predictor(s).")
            list_of_predictors = selected

        print(f"\n  Final selected predictors ({len(list_of_predictors)}): {list_of_predictors}")
        print("=" * 62 + "\n")

        if not list_of_predictors:
            print("No predictors were selected. Returning None.")
            return None

    # Add constant
    if add_constant:
        dataframe = sm.add_constant(dataframe)
    
    # Scale the predictors, if requested
    if scale_predictors:
        # Show the mean and standard deviation of each predictor
        print("\nMean of each predictor:")
        print(dataframe[list_of_predictors].mean())
        print("\nStandard deviation of each predictor:")
        print(dataframe[list_of_predictors].std())
        
        # Scale predictors
        dataframe[list_of_predictors] = StandardScaler().fit_transform(dataframe[list_of_predictors])
    
    # Create linear regression model
    if add_constant:
        model = sm.OLS(dataframe[outcome_variable], dataframe[['const'] + list_of_predictors])
    else:
        model = sm.OLS(dataframe[outcome_variable], dataframe[list_of_predictors])
    model_res = model.fit()
    model_summary = model_res.summary()

    # Show the F-statistic and p-value of the model, along with some text to help interpret it
    if show_help:
        print("\nThe F-statistic tests the null hypothesis that all of the regression coefficients are equal to zero. "
              "The alternative hypothesis is that at least one of the regression coefficients is not equal to zero.")
        print("\nF-statistic value:", model_res.fvalue)
        print("F-statistic p-value:", model_res.f_pvalue)
        if model_res.f_pvalue < 0.05:
            print("The F-statistic is statistically significant, meaning that we can reject the null hypothesis.")
        else:
            print("The F-statistic is not statistically significant, meaning that we cannot reject the null hypothesis.")
    
    # If requested, show diagnostic plots
    if show_diagnostic_plots_for_each_predictor:
        for variable in list_of_predictors:
            fig = plt.figure(figsize=(12, 8))
            fig = sm.graphics.plot_regress_exog(model_res, variable, fig=fig)
    
    # If requested, show help text
    if show_help:
        print(
            "Quick guide on accessing output of ConductLinearRegressionAnalysis function:",
            "\nThe ouput of the ConductLinearRegressionAnalysis function is a dictionary containing the regression results, a test dataset of predictors, and a test dataset of outcomes.",
            "\n\t--To access the linear regression model, use the 'Fitted Model' key."
            "\n\t--To view the model's statistical summary, use the 'Model Summary' key."
        )
    
    # Create dictionary of objects to return
    dict_return = {
        "Fitted Model": model_res,
        "Model Summary": model_summary
    }
    return dict_return

