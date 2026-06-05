# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Declare function
def CreateLogisticRegressionModel(dataframe,
                                  outcome_variable,
                                  list_of_predictor_variables,
                                  scale_predictor_variables=False,
                                  # Output arguments
                                  print_peak_to_peak_range_of_each_predictor=False,
                                  test_size=0.2,
                                  show_classification_plot=True,
                                  lambda_for_regularization=0.001,
                                  max_iterations=1000,
                                  random_seed=412,
                                  # Output arguments
                                  print_model_training_performance=False,
                                  # MSE/accuracy comparison plot arguments
                                  plot_training_and_test_mse=True,
                                  training_bar_color="#3a86ff",
                                  test_bar_color="#b0170c",
                                  figure_size_for_mse_comparison_plot=(7, 5),
                                  title_for_mse_comparison_plot=None,
                                  subtitle_for_mse_comparison_plot=None,
                                  caption_for_mse_comparison_plot=None,
                                  title_y_indent_for_mse_comparison_plot=1.10,
                                  subtitle_y_indent_for_mse_comparison_plot=1.05,
                                  caption_y_indent_for_mse_comparison_plot=-0.15,
                                  x_indent_for_mse_comparison_plot=-0.115):
    """
    Train, evaluate, and visualize a logistic regression model for binary classification.

    This function utilizes scikit-learn's `LogisticRegression` to model the 
    probability of a discrete outcome (such as success/failure or yes/no) given 
    a set of independent predictor variables. It handles automated data cleaning, 
    optional feature scaling, and generates a confusion matrix heatmap to assess 
    classification accuracy and error patterns.

    Logistic regression is essential for:
      * Predicting the likelihood of customer conversion or subscription renewal
      * Assessing the probability of default in credit risk analysis
      * Identifying the drivers behind binary choices in consumer behavior
      * Classifying medical or technical reports into specific categories (e.g., critical/standard)
      * Forecasting the outcome of competitive bids or project approvals
      * Evaluating the impact of different features on a categorical classification
      * Building efficient baseline classifiers for machine learning pipelines

    The function provides control over regularization strength and iteration 
    limits, ensuring model convergence even with complex datasets. It 
    automatically handles complete case analysis by removing rows with missing or 
    infinite values and offers integrated visualization of the accuracy score 
    and confusion results.

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing the training and testing data.
    outcome_variable
        The name of the categorical target column to be predicted.
    list_of_predictor_variables
        A list of column names used as features for the classification.
    scale_predictor_variables
        If True, scales the features using `StandardScaler` prior to training. 
        Defaults to False.
    print_peak_to_peak_range_of_each_predictor
        If True, prints the statistical range of each predictor column to 
        monitor data dispersion. Defaults to False.
    test_size
        The proportion of the dataset used for testing. Defaults to 0.2.
    show_classification_plot
        Whether to display a heatmap of the confusion matrix labeled with 
        the overall accuracy score. Defaults to True.
    lambda_for_regularization
        The regularization parameter. Note: internally mapped to C = 1 - lambda. 
        Lower values increase regularization strength. Defaults to 0.001.
    max_iterations
        The maximum number of iterations allowed for the solver to converge.
        Defaults to 1000.
    random_seed
        Controls the randomness of the data split and solver initialization.
        Defaults to 412.
    print_model_training_performance
        If True, prints training accuracy and test accuracy after fitting.
        Defaults to False.
    plot_training_and_test_mse
        Whether to render a bar chart comparing training accuracy and test
        accuracy. Defaults to True.
    training_bar_color, test_bar_color
        Bar colors for the training and test bars. Defaults to blue / red.
    figure_size_for_mse_comparison_plot
        Dimensions (width, height) for the comparison chart. Defaults to (7, 5).
    title_for_mse_comparison_plot, subtitle_for_mse_comparison_plot, caption_for_mse_comparison_plot
        Text elements for the comparison chart. Sensible defaults are used when None.
    title_y_indent_for_mse_comparison_plot, subtitle_y_indent_for_mse_comparison_plot, caption_y_indent_for_mse_comparison_plot, x_indent_for_mse_comparison_plot
        Coordinate offsets for text placement in the comparison chart.

    Returns
    -------
    sklearn.linear_model.LogisticRegression or dict
        If `scale_predictor_variables` is False, returns the fitted 
        LogisticRegression model. If True, returns a dictionary containing 
        both the 'model' and the 'scaler' object.

    Examples
    --------
    # Create a basic logistic model to predict customer churn
    model = CreateLogisticRegressionModel(
        df, 
        outcome_variable='is_churn', 
        list_of_predictor_variables=['tenure', 'monthly_spend']
    )

    # Build a scaled model with custom regularization and iteration limits
    results = CreateLogisticRegressionModel(
        credit_df,
        outcome_variable='default_status',
        list_of_predictor_variables=['income', 'debt_ratio', 'age'],
        scale_predictor_variables=True,
        lambda_for_regularization=0.01,
        max_iterations=2000
    )

    """
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.dropna(inplace=True)
    # print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
    # Scale the predictors, if requested
    if scale_predictor_variables:
        # Scale predictors
        scaler = StandardScaler()
        dataframe[list_of_predictor_variables] = scaler.fit_transform(dataframe[list_of_predictor_variables])
        
    # Show the peak-to-peak range of each predictor
    if print_peak_to_peak_range_of_each_predictor:
        print("\nPeak-to-peak range of each predictor:")
        print(np.ptp(dataframe[list_of_predictor_variables], axis=0))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create logistic regression model
    model = LogisticRegression(
        max_iter=max_iterations, 
        random_state=random_seed,
        C=1-lambda_for_regularization,
        fit_intercept=True
    )
    
    # Train the model using the training sets and show fitting summary
    model.fit(train[list_of_predictor_variables], train[outcome_variable])
    print(f"\nNumber of iterations completed: {model.n_iter_}")
    
    # Show parameters of the model
    b_norm = model.intercept_
    w_norm = model.coef_
    print(f"\nModel parameters:    w: {w_norm}, b:{b_norm}")
    
    # Predict on training and test sets
    train['Predicted'] = model.predict(train[list_of_predictor_variables])
    test['Predicted'] = model.predict(test[list_of_predictor_variables])

    # Compute training and test accuracy
    training_accuracy = metrics.accuracy_score(train[outcome_variable], train['Predicted'])
    score = model.score(test[list_of_predictor_variables], test[outcome_variable])

    # Print training and test accuracy
    if print_model_training_performance:
        print('Training Accuracy:', training_accuracy)
        print('Test Accuracy:', score)

    # Print the confusion matrix
    confusion_matrix = metrics.confusion_matrix(
        test[outcome_variable], 
        test['Predicted']
    )
    if show_classification_plot:
        plt.figure(figsize=(9,9))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt=".3f", 
            linewidths=.5, 
            square=True, 
            cmap='Blues_r'
        )
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15)
        plt.show()
    else:
        print("Confusion matrix:")
        print(confusion_matrix)
        
    # Plot training vs. test accuracy comparison
    if plot_training_and_test_mse:
        import textwrap as _tw
        chart_title = title_for_mse_comparison_plot or "Training vs. Test Accuracy"
        chart_subtitle = subtitle_for_mse_comparison_plot or "Compares model accuracy on the training and test datasets."

        fig, ax = plt.subplots(figsize=figure_size_for_mse_comparison_plot)
        bar_values = [training_accuracy, score]
        bar_labels = ['Training', 'Test']
        bar_colors = [training_bar_color, test_bar_color]
        bars = ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.8, width=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                    '{:.4f}'.format(height), ha='center', va='bottom', fontsize=10, color='#262626')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_visible(False)
        ax.tick_params(which='major', labelsize=9, color='#666666')
        ax.yaxis.set_ticks([])
        plt.subplots_adjust(top=0.85)
        ax.text(x=x_indent_for_mse_comparison_plot, y=title_y_indent_for_mse_comparison_plot,
                s=chart_title, fontsize=14, color="#262626", transform=ax.transAxes)
        ax.text(x=x_indent_for_mse_comparison_plot, y=subtitle_y_indent_for_mse_comparison_plot,
                s=chart_subtitle, fontsize=11, color="#666666", transform=ax.transAxes)
        if caption_for_mse_comparison_plot is not None:
            ax.text(x=x_indent_for_mse_comparison_plot, y=caption_y_indent_for_mse_comparison_plot,
                    s=_tw.fill(caption_for_mse_comparison_plot, 80, break_long_words=False),
                    fontsize=8, color="#666666", transform=ax.transAxes)
        plt.show()
        plt.clf()

    # Return the model
    if scale_predictor_variables:
        dict_return = {
            'model': model,
            'scaler': scaler
        }
        return(dict_return)
    else:
        return(model)

