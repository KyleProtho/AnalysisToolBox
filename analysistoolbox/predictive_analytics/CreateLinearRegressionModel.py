# Load pacakges
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import textwrap

# Declare function
def CreateLinearRegressionModel(dataframe,
                                outcome_variable,
                                list_of_predictor_variables,
                                # Model parameters
                                scale_variables=False,
                                test_size=0.2,
                                fit_intercept=True,
                                random_seed=412,
                                variable_selection='none',
                                selection_limit=4,
                                selection_p_threshold=0.05,
                                # Output arguments
                                print_peak_to_peak_range_of_each_predictor=False,
                                print_model_training_performance=False,
                                # All plot arguments
                                data_source_for_plot=None,
                                # Model performance plot arguments
                                plot_model_test_performance=True,
                                dot_fill_color="#999999",
                                line_color=None,
                                figure_size_for_model_test_performance_plot=(8, 6),
                                title_for_model_test_performance_plot="Model Performance",
                                subtitle_for_model_test_performance_plot="The predicted values vs. the actual values in the test dataset.",
                                caption_for_model_test_performance_plot=None,
                                title_y_indent_for_model_test_performance_plot=1.10,
                                subtitle_y_indent_for_model_test_performance_plot=1.05,
                                caption_y_indent_for_model_test_performance_plot=-0.215,
                                x_indent_for_model_test_performance_plot=-0.115,
                                # Feature importance plot arguments
                                plot_feature_importance=True,
                                top_n_to_highlight=3,
                                highlight_color="#b0170c",
                                fill_transparency=0.8,
                                figure_size_for_feature_importance_plot=(8, 6),
                                title_for_feature_importance_plot="Feature Importance",
                                subtitle_for_feature_importance_plot="Shows the predictive power of each feature in the model.",
                                caption_for_feature_importance_plot=None,
                                title_y_indent_for_feature_importance_plot=1.15,
                                subtitle_y_indent_for_feature_importance_plot=1.1,
                                caption_y_indent_for_feature_importance_plot=-0.15,
                                # Performance comparison plot arguments
                                plot_training_and_test_performance=True,
                                training_bar_color="#3a86ff",
                                test_bar_color="#b0170c",
                                figure_size_for_performance_comparison_plot=(7, 5),
                                title_for_performance_comparison_plot="Training vs. Test MSE",
                                subtitle_for_performance_comparison_plot="Compares model error on the training and test datasets.",
                                caption_for_performance_comparison_plot=None,
                                title_y_indent_for_performance_comparison_plot=1.10,
                                subtitle_y_indent_for_performance_comparison_plot=1.05,
                                caption_y_indent_for_performance_comparison_plot=-0.15,
                                x_indent_for_performance_comparison_plot=-0.115):
    """
    Train, evaluate, and visualize a multiple linear regression model.

    This function utilizes scikit-learn's `LinearRegression` to model the linear 
    relationship between a dependent outcome variable and one or more independent 
    predictor variables. It handles automated data cleaning, optional feature 
    scaling, and provides comprehensive diagnostic visualizations to assess model 
    accuracy and feature influence.

    Linear regression is essential for:
      * Estimating the impact of marketing spend on sales revenue
      * Analyzing the relationship between macroeconomic indicators and asset prices
      * Predicting infrastructure or energy demand based on seasonal variables
      * Assessing the influence of demographic factors on social or health outcomes
      * Identifying key operational drivers of efficiency in manufacturing processes
      * Modeling the sensitivity of output variables to changes in input parameters
      * Establishing baseline predictive models for continuous data analysis

    The function offers integrated performance evaluation (MSE, R-squared) and 
    generates regression plots to visualize the fit between predicted and 
    actual values. It also produces a "Feature Importance" chart based on beta 
    coefficients, helping analysts identify the strongest drivers within their 
    data.

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing both predictor and outcome variables.
    outcome_variable
        The name of the target column (dependent variable) to be predicted.
    list_of_predictor_variables
        A list of column names (independent variables) used to train the model.
    scale_variables
        If True, scales the predictor variables using `StandardScaler` prior 
        to modeling. Recommended for variables with vastly different units. 
        Defaults to False.
    test_size
        The proportion of the dataset used for testing. Set to 0 to train on 
        the full dataset. Defaults to 0.2.
    fit_intercept
        Whether to calculate the intercept for this model. If False, the 
        intercept will be set to 0.0. Defaults to True.
    random_seed
        Controls the randomness of the train-test split for reproducibility.
        Defaults to 412.
    variable_selection
        Stepwise variable selection method applied to the training set before
        fitting the final model. Options:
          * 'none'     — No selection; use all predictors in
            ``list_of_predictor_variables``.
          * 'forward'  — Start from the null model (intercept only) and greedily
            add the predictor that most lowers AIC (computed from training RSS),
            stopping when no addition helps or ``selection_limit`` is reached.
          * 'backward' — Start with all predictors and iteratively remove the one
            with the highest p-value until all remaining p-values fall below
            ``selection_p_threshold``.
          * 'mixed'    — Forward steps add variables by AIC improvement; after each
            addition, any in-model predictor whose p-value exceeds
            ``selection_p_threshold`` is removed. Continues until stable.
        Defaults to 'none'.
    selection_limit
        Hard cap on the number of predictors kept by 'forward' or 'mixed'
        selection. Defaults to 4.
    selection_p_threshold
        P-value threshold used by 'backward' and 'mixed' selection. Predictors
        with p-value at or above this threshold are candidates for removal.
        Defaults to 0.05.
    print_peak_to_peak_range_of_each_predictor
        If True, prints the statistical range of each predictor column to 
        the console. Defaults to False.
    print_model_training_performance
        If True, prints model accuracy metrics (MSE and R-squared Variance 
        Score). Defaults to False.
    data_source_for_plot
        Source citation string displayed in the caption of all generated plots. 
        Defaults to None.
    plot_model_test_performance
        Whether to generate a regression plot showing Predicted vs. Actual 
        values on the test set. Defaults to True.
    dot_fill_color, line_color
        Aesthetic settings for the regression performance plot.
    figure_size_for_model_test_performance_plot
        Dimensions (width, height) for the performance visualization. 
        Defaults to (8, 6).
    title_for_model_test_performance_plot, subtitle_for_model_test_performance_plot, caption_for_model_test_performance_plot
        Text elements for the performance visualization.
    title_y_indent_for_model_test_performance_plot, subtitle_y_indent_for_model_test_performance_plot, caption_y_indent_for_model_test_performance_plot, x_indent_for_model_test_performance_plot
        Coordinate offsets for text placement in the performance plot.
    plot_feature_importance
        Whether to generate a horizontal bar chart showing the magnitude of 
        calculated beta coefficients. Defaults to True.
    top_n_to_highlight
        The number of top influential features to color differently in the 
        importance plot. Defaults to 3.
    highlight_color, fill_transparency
        Aesthetic settings for the feature importance bars.
    figure_size_for_feature_importance_plot
        Dimensions (width, height) for the importance visualization. 
        Defaults to (8, 6).
    title_for_feature_importance_plot, subtitle_for_feature_importance_plot, caption_for_feature_importance_plot
        Text elements for the feature importance visualization.
    title_y_indent_for_feature_importance_plot, subtitle_y_indent_for_feature_importance_plot, caption_y_indent_for_feature_importance_plot
        Coordinate offsets for text placement in the importance plot.
    plot_training_and_test_performance
        Whether to generate a bar chart comparing MSE on the training set versus
        the test set. A large gap indicates overfitting. Defaults to True.
    training_bar_color
        Fill color for the training MSE bar. Defaults to "#3a86ff".
    test_bar_color
        Fill color for the test MSE bar. Defaults to "#b0170c".
    figure_size_for_performance_comparison_plot
        Dimensions (width, height) for the MSE comparison chart. Defaults to (7, 5).
    title_for_performance_comparison_plot, subtitle_for_performance_comparison_plot, caption_for_performance_comparison_plot
        Text elements for the MSE comparison chart.
    title_y_indent_for_performance_comparison_plot, subtitle_y_indent_for_performance_comparison_plot, caption_y_indent_for_performance_comparison_plot, x_indent_for_performance_comparison_plot
        Coordinate offsets for text placement in the MSE comparison chart.

    Returns
    -------
    sklearn.linear_model.LinearRegression or sklearn.pipeline.Pipeline
        The fitted linear model object. If `scale_variables` is True, a Pipeline 
        containing the scaler and regressor is returned.

    Examples
    --------
    # Create a basic regression model to forecast revenue
    model = CreateLinearRegressionModel(
        df, 
        outcome_variable='Revenue', 
        list_of_predictor_variables=['AdSpend', 'Followers', 'Season']
    )

    # Build a scaled model with high-performance reporting and custom colors
    model = CreateLinearRegressionModel(
        df,
        outcome_variable='HousePrice',
        list_of_predictor_variables=['SqFt', 'Bedrooms', 'Age'],
        scale_variables=True,
        print_model_training_performance=True,
        highlight_color='teal',
        line_color='darkorange'
    )

    """
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Replace inf with nan, and drop rows with nan
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.dropna(inplace=True)
    # print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
    # Scale the predictors, if requested
    if scale_variables:
        # Scale predictors
        scaler = StandardScaler()
        dataframe[list_of_predictor_variables] = scaler.fit_transform(dataframe[list_of_predictor_variables])
        
    # Show the peak-to-peak range of each predictor
    if print_peak_to_peak_range_of_each_predictor:
        print("\nPeak-to-peak range of each predictor:")
        print(np.ptp(dataframe[list_of_predictor_variables], axis=0))
    
    # Split dataframe into training and test sets
    if test_size > 0:
        train, test = train_test_split(
            dataframe, 
            test_size=test_size,
            random_state=random_seed
        )
    else:
        train = dataframe.copy()
        test = dataframe.copy()

    # Variable selection (forward / backward / mixed), run on training data
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

        n_train = len(train)
        y_train = train[outcome_variable].values

        # Helper: AIC from training RSS for a given variable list
        def _aic(var_list):
            _m = linear_model.LinearRegression(fit_intercept=fit_intercept)
            _m.fit(train[var_list], y_train)
            rss = np.sum((y_train - _m.predict(train[var_list])) ** 2)
            if rss <= 0:
                return -np.inf
            k = len(var_list) + (1 if fit_intercept else 0)
            return n_train * np.log(rss / n_train) + 2 * k

        # Helper: OLS p-values for each predictor (excluding intercept)
        def _pvalues(var_list):
            from scipy import stats as _stats
            X = train[var_list].values
            n, p = X.shape
            _m = linear_model.LinearRegression(fit_intercept=fit_intercept)
            _m.fit(X, y_train)
            rss = np.sum((y_train - _m.predict(X)) ** 2)
            df_resid = n - p - (1 if fit_intercept else 0)
            if df_resid <= 0:
                return {v: 1.0 for v in var_list}
            sigma_sq = rss / df_resid
            X_aug = np.column_stack([np.ones(n), X]) if fit_intercept else X
            try:
                cov = sigma_sq * np.linalg.inv(X_aug.T @ X_aug)
            except np.linalg.LinAlgError:
                return {v: 1.0 for v in var_list}
            se = np.sqrt(np.diag(cov))
            pred_se = se[1:] if fit_intercept else se
            t_stats = _m.coef_ / pred_se
            pvals = 2 * (1 - _stats.t.cdf(np.abs(t_stats), df=df_resid))
            return dict(zip(var_list, pvals))

        # Null model AIC baseline
        if fit_intercept:
            null_rss = np.sum((y_train - y_train.mean()) ** 2)
            null_aic = n_train * np.log(null_rss / n_train) + 2
        else:
            null_rss = np.sum(y_train ** 2)
            null_aic = n_train * np.log(null_rss / n_train) if null_rss > 0 else -np.inf

        # ── FORWARD SELECTION ────────────────────────────────────────
        if variable_selection == 'forward':
            remaining = list_of_predictor_variables.copy()
            selected = []
            current_aic = null_aic
            print(f"\n  Null model AIC (intercept only): {current_aic:.4f}")
            print(f"  Candidates:  {remaining}")
            print(f"  Limit:       {selection_limit}\n")

            while remaining and len(selected) < selection_limit:
                best_aic, best_var = current_aic, None
                for var in remaining:
                    cand_aic = _aic(selected + [var])
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
            list_of_predictor_variables = selected

        # ── BACKWARD SELECTION ───────────────────────────────────────
        elif variable_selection == 'backward':
            current_preds = list_of_predictor_variables.copy()
            print(f"\n  Starting predictors: {current_preds}")
            print(f"  P-value threshold:   {selection_p_threshold}\n")
            step = 0

            while len(current_preds) > 0:
                pvals = _pvalues(current_preds)
                worst_var = max(pvals, key=pvals.get)
                worst_p = pvals[worst_var]
                if worst_p <= selection_p_threshold:
                    print(f"  All remaining predictors have p-value <= {selection_p_threshold}. Stopping.")
                    break
                step += 1
                current_preds.remove(worst_var)
                label = current_preds if current_preds else ['(none)']
                print(f"  Step {step}: Removed '{worst_var}' | p-value: {worst_p:.4f} | Remaining: {label}")

            list_of_predictor_variables = current_preds

        # ── MIXED SELECTION ──────────────────────────────────────────
        elif variable_selection == 'mixed':
            remaining = list_of_predictor_variables.copy()
            selected = []
            current_aic = null_aic
            print(f"\n  Null model AIC (intercept only): {current_aic:.4f}")
            print(f"  Candidates: {remaining}")
            print(f"  Limit: {selection_limit}  |  P-value threshold: {selection_p_threshold}\n")

            step = 0
            seen_states = set()
            max_iter = (len(list_of_predictor_variables) + 1) * 4

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
                        cand_aic = _aic(selected + [var])
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
                    pvals = _pvalues(selected)
                    worst_var = max(pvals, key=pvals.get)
                    worst_p = pvals[worst_var]
                    if worst_p > selection_p_threshold:
                        selected.remove(worst_var)
                        remaining.append(worst_var)
                        step += 1
                        print(f"  Step {step} [-]: Removed '{worst_var}' | p-value: {worst_p:.4f}")
                        current_aic = _aic(selected) if selected else null_aic
                        backward_taken = True

                if not forward_taken and not backward_taken:
                    print("  Converged: no further additions or removals improve the model.")
                    break

            if len(selected) == selection_limit and remaining:
                print(f"\n  Stopped: reached the limit of {selection_limit} predictor(s).")
            list_of_predictor_variables = selected

        print(f"\n  Final selected predictors ({len(list_of_predictor_variables)}): {list_of_predictor_variables}")
        print("=" * 62 + "\n")

        if not list_of_predictor_variables:
            print("No predictors were selected. Returning None.")
            return None

    # Create linear regression object
    if scale_variables:
        model = make_pipeline(StandardScaler(),
                              linear_model.LinearRegression(
                                  fit_intercept=fit_intercept,
                              )
        )
    else:
        model = linear_model.LinearRegression(
            fit_intercept=fit_intercept,
        )
    
    # Train the model using the training sets and show fitting summary
    model.fit(X=train[list_of_predictor_variables], 
              y=train[outcome_variable])
    
    # Show number of iterations and weight updates
    if scale_variables:
        regressor = model['linearregression']
    else:
        regressor = model
    
    # # Show parameters of the model
    # b_norm = model.intercept_
    # w_norm = model.coef_
    # print(f"\nModel parameters:    w: {w_norm}, b:{b_norm}")
    
    # Add predictions to training and test sets
    train['Predicted'] = model.predict(train[list_of_predictor_variables])
    test['Predicted'] = model.predict(test[list_of_predictor_variables])

    # Compute training and test MSE
    training_mse = metrics.mean_squared_error(train[outcome_variable], train['Predicted'])
    test_mse = metrics.mean_squared_error(test[outcome_variable], test['Predicted'])

    # Show mean squared error if outcome is numerical
    if print_model_training_performance:
        print('Training MSE:', training_mse)
        print('Test MSE:', test_mse)
        print('Variance Score:', metrics.r2_score(test[outcome_variable], test['Predicted']))
        print("Note: A variance score of 1 is perfect prediction and 0 means that there is no linear relationship between X and Y.")
        
    # Plot predicted and observed outputs if requested
    if plot_model_test_performance:
        # Set the size of the plot
        plt.figure(figsize=figure_size_for_model_test_performance_plot)
        
        # Generate a scatterplot of the predicted vs. observed outcome
        ax = sns.regplot(
            data=test,
            x=outcome_variable,
            y='Predicted',
            marker='o',
            scatter_kws={
                'color': dot_fill_color,
                'alpha': 0.5,
                # 'linewidth': 0.5,
                'edgecolor': dot_fill_color
            },
            lowess=True,
            line_kws={'color': line_color}
        )
        
        # Add a "perfect prediction" line
        plt.plot(
            test[outcome_variable], 
            test[outcome_variable], 
            color='black', 
            alpha=0.35, 
            linewidth=0.5, 
            linestyle='--'
        )
        
        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        
        # Format tick labels to be Arial, size 9, and color #666666
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
        )
        
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent_for_model_test_performance_plot,
            y=title_y_indent_for_model_test_performance_plot,
            s=title_for_model_test_performance_plot,
            # fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent_for_model_test_performance_plot,
            y=subtitle_y_indent_for_model_test_performance_plot,
            s=subtitle_for_model_test_performance_plot,
            # fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
        
        # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
        ax.yaxis.set_label_coords(-0.1, 0.99)
        ax.yaxis.set_label_text(
            'Predicted',
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right',
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.99, -0.1)
        ax.xaxis.set_label_text(
            textwrap.fill(outcome_variable, 30, break_long_words=False),
            # fontname="Arial",
            fontsize=10,
            color="#666666",
            ha='right',
        )
        
        # Add a word-wrapped caption if one is provided
        if caption_for_model_test_performance_plot != None or data_source_for_plot != None:
            # Create starting point for caption
            wrapped_caption = ""
            
            # Add the caption to the plot, if one is provided
            if caption_for_model_test_performance_plot != None:
                # Word wrap the caption without splitting words
                wrapped_caption = textwrap.fill(caption_for_model_test_performance_plot, 130, break_long_words=False)
                
            # Add the data source to the caption, if one is provided
            if data_source_for_plot != None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            
            # Add the caption to the plot
            ax.text(
                x=x_indent_for_model_test_performance_plot,
                y=caption_y_indent_for_model_test_performance_plot,
                s=wrapped_caption,
                # fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
            
            # Show the plot
            plt.show()
    
    # Plot feature importance if requested
    if plot_feature_importance:
        # Get importance
        importance = regressor.coef_
        
        # Create lists to store feature names and feature importance
        feature_names = []
        feature_importance = []
        
        # Store feature names and feature importance in lists
        for i,v in enumerate(importance):
            feature_names.append(i)
            feature_importance.append(v)
        # Create dataframe of feature importance
        data_feauture_importance = pd.DataFrame(
            data={
                'Feature': model.feature_names_in_,
                'Importance': regressor.coef_
            }
        )
        
        # Sort dataframe by importance
        data_feauture_importance = data_feauture_importance.sort_values(by='Importance', ascending=False)
        
        # Highlight top n features
        data_feauture_importance['Highlighted'] = np.where(
            data_feauture_importance['Feature'].isin(data_feauture_importance['Feature'].head(top_n_to_highlight)),
            True,
            False
        )
        
        # Plot feature importance with seaborn, using a horizontal barplot
        plt.figure(figsize=figure_size_for_feature_importance_plot)
        ax = sns.barplot(
            data=data_feauture_importance,
            x='Importance',
            y='Feature',
            hue='Highlighted',
            palette={True: highlight_color, False: "#b8b8b8"},
            alpha=fill_transparency,
            dodge=False
        )
        
        # Remove the legend
        ax.legend_.remove()
        
        # Format and wrap y axis tick labels using textwrap
        y_tick_labels = ax.get_yticklabels()
        wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 50)) for label in y_tick_labels]
        ax.set_yticklabels(
            wrapped_y_tick_labels, 
            fontsize=10, 
            # fontname="Arial", 
            color="#262626"
        )
        
        # Remove a-axis tick labels
        ax.get_xaxis().set_ticks([])
        
        # Format x-axis label
        ax.set_xlabel(
            "Beta Coefficent", 
            fontsize=10, 
            # fontname="Arial", 
            color="#262626"
        )
        
        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('#b8b8b8')
        ax.spines['bottom'].set_visible(False)
        
        # Add data labels
        for container in ax.containers:
            ax.bar_label(
                container, 
                fmt='%.3f', 
                label_type='edge', 
                padding=5,
                fontsize=10, 
                # fontname="Arial", 
                color="#262626"
            )
        
        # Add space between the title and the plot
        plt.subplots_adjust(top=0.85)
        
        # Set the x indent of the plot titles and captions
        # Get longest y tick label
        longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
        if len(longest_y_tick_label) >= 30:
            x_indent = -0.3
        else:
            x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)
        
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent,
            y=title_y_indent_for_feature_importance_plot,
            s=title_for_feature_importance_plot,
            # fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent_for_feature_importance_plot,
            s=subtitle_for_feature_importance_plot,
            # fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
        
        # Add a word-wrapped caption if one is provided
        if caption_for_feature_importance_plot != None or data_source_for_plot != None:
            # Create starting point for caption
            wrapped_caption = ""
            
            # Add the caption to the plot, if one is provided
            if caption_for_feature_importance_plot != None:
                # Word wrap the caption without splitting words
                wrapped_caption = textwrap.fill(caption_for_feature_importance_plot, 110, break_long_words=False)
                
            # Add the data source to the caption, if one is provided
            if data_source_for_plot != None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            
            # Add the caption to the plot
            ax.text(
                x=x_indent,
                y=caption_y_indent_for_feature_importance_plot,
                s=wrapped_caption,
                # fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
            
        # Show the plot
        plt.show()
        plt.clf()
    
    # Plot training vs. test MSE comparison if requested
    if plot_training_and_test_performance:
        fig, ax = plt.subplots(figsize=figure_size_for_performance_comparison_plot)

        mse_values = [training_mse, test_mse]
        bar_labels = ['Training', 'Test']
        bar_colors = [training_bar_color, test_bar_color]

        bars = ax.bar(
            bar_labels,
            mse_values,
            color=bar_colors,
            alpha=0.8,
            width=0.5
        )

        # Add data labels above each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:,.4f}',
                ha='center',
                va='bottom',
                fontsize=10,
                color='#262626'
            )

        # Remove top and right spines, style remaining spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_visible(False)

        # Format tick labels
        ax.tick_params(which='major', labelsize=9, color='#666666')
        ax.yaxis.set_ticks([])

        # Add space for title and subtitle
        plt.subplots_adjust(top=0.85)

        # Title
        ax.text(
            x=x_indent_for_performance_comparison_plot,
            y=title_y_indent_for_performance_comparison_plot,
            s=title_for_performance_comparison_plot,
            fontsize=14,
            color='#262626',
            transform=ax.transAxes
        )

        # Subtitle
        ax.text(
            x=x_indent_for_performance_comparison_plot,
            y=subtitle_y_indent_for_performance_comparison_plot,
            s=subtitle_for_performance_comparison_plot,
            fontsize=11,
            color='#666666',
            transform=ax.transAxes
        )

        # Caption
        if caption_for_performance_comparison_plot is not None or data_source_for_plot is not None:
            wrapped_caption = ""
            if caption_for_performance_comparison_plot is not None:
                wrapped_caption = textwrap.fill(caption_for_performance_comparison_plot, 130, break_long_words=False)
            if data_source_for_plot is not None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            ax.text(
                x=x_indent_for_performance_comparison_plot,
                y=caption_y_indent_for_performance_comparison_plot,
                s=wrapped_caption,
                fontsize=8,
                color='#666666',
                transform=ax.transAxes
            )

        plt.show()
        plt.clf()

    # Return the model
    return model

