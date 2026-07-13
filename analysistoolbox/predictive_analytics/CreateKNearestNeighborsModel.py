# Load packages
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import textwrap


# Declare function
def CreateKNearestNeighborsModel(dataframe,
                                 outcome_variable,
                                 list_of_predictor_variables,
                                 # Model type
                                 model_type='regressor',
                                 # Model parameters
                                 n_neighbors=5,
                                 weights='uniform',
                                 algorithm='auto',
                                 leaf_size=30,
                                 p=2,
                                 radius=1.0,
                                 scale_variables=True,
                                 test_size=0.2,
                                 random_seed=412,
                                 # Output arguments
                                 print_model_training_performance=False,
                                 # All plot arguments
                                 data_source_for_plot=None,
                                 # Model performance plot arguments
                                 plot_model_test_performance=True,
                                 dot_fill_color="#999999",
                                 line_color=None,
                                 heatmap_color_palette="Blues",
                                 figure_size_for_model_test_performance_plot=(8, 6),
                                 title_for_model_test_performance_plot="Model Performance",
                                 subtitle_for_model_test_performance_plot="The predicted values vs. the actual values in the test dataset.",
                                 caption_for_model_test_performance_plot=None,
                                 title_y_indent_for_model_test_performance_plot=1.09,
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
                                 subtitle_for_feature_importance_plot="Shows how much each predictor affects model accuracy when its values are randomly shuffled.",
                                 caption_for_feature_importance_plot=None,
                                 title_y_indent_for_feature_importance_plot=1.15,
                                 subtitle_y_indent_for_feature_importance_plot=1.1,
                                 caption_y_indent_for_feature_importance_plot=-0.15,
                                 # Performance comparison plot arguments
                                 plot_training_and_test_performance=True,
                                 training_bar_color="#3a86ff",
                                 test_bar_color="#b0170c",
                                 figure_size_for_performance_comparison_plot=(7, 5),
                                 title_for_performance_comparison_plot=None,
                                 subtitle_for_performance_comparison_plot=None,
                                 caption_for_performance_comparison_plot=None,
                                 title_y_indent_for_performance_comparison_plot=1.10,
                                 subtitle_y_indent_for_performance_comparison_plot=1.05,
                                 caption_y_indent_for_performance_comparison_plot=-0.15,
                                 x_indent_for_performance_comparison_plot=-0.115):
    """
    Train, evaluate, and visualize a K-Nearest Neighbors (KNN) model.

    This function builds a K-Nearest Neighbors model using scikit-learn,
    supporting regression, classification, and radius-based variants. It handles
    automated data cleaning, optional feature scaling, and provides diagnostic
    visualizations for model accuracy and feature influence via permutation
    importance.

    K-Nearest Neighbors is useful for:
      * Recommending products or content based on similarity to past user behavior
      * Classifying customer segments based on behavioral or demographic profiles
      * Estimating property values from comparable recent sales
      * Detecting anomalous observations by their distance from known patterns
      * Imputing missing values using outcomes from similar observations
      * Establishing a flexible baseline predictive model for tabular datasets
      * Fraud detection by identifying transactions that look unlike normal ones

    Teaching Note
    -------------
    KNN is a "lazy learner" — it memorizes the entire training dataset rather
    than learning an explicit mathematical formula. When asked to predict a new
    observation, it finds the *k* most similar (nearest) training examples and
    either averages their outcomes (regression) or takes a majority vote
    (classification).

    This simplicity is both its strength and its limitation. KNN requires no
    assumptions about the underlying data distribution, making it flexible. But
    because it computes distances between observations to find neighbors, it is
    highly sensitive to two things:

    1. **Feature scale**: A variable measured in thousands (e.g., income) will
       dominate distance calculations over a variable measured in ones (e.g., number
       of children). Always scale predictor variables before using KNN — this
       function enables ``scale_variables=True`` by default for this reason.

    2. **Number of predictors**: As more variables are added, the geometric space
       becomes increasingly sparse. Every point starts to look equally far from
       every other point, so "nearest neighbors" become less meaningful. This is
       known as the "curse of dimensionality." In practice, 4 or fewer well-chosen
       predictors often outperform a model with many noisy or redundant ones.

    Because KNN has no analytical formula (unlike linear regression), feature
    importance is measured via permutation importance: each predictor is randomly
    shuffled one at a time, and the drop in model accuracy shows how much the
    model depended on that feature.

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing both predictor and outcome variables.
    outcome_variable
        The name of the target column (dependent variable) to be predicted.
    list_of_predictor_variables
        A list of column names (independent variables) used to train the model.
    model_type
        The KNN variant to use. One of:
          * ``'regressor'`` (default) — predicts a continuous numeric outcome
            using the k nearest neighbors.
          * ``'classifier'`` — predicts a categorical class label using the k
            nearest neighbors.
          * ``'radius_regressor'`` — regression using all neighbors within a
            fixed radius.
          * ``'radius_classifier'`` — classification using all neighbors within
            a fixed radius.
    n_neighbors
        The number of nearest neighbors to use for prediction. Applies to
        ``'regressor'`` and ``'classifier'`` model types. Defaults to 5.
    weights
        How neighbor contributions are weighted. ``'uniform'`` gives equal weight
        to all neighbors; ``'distance'`` gives closer neighbors more influence.
        Defaults to ``'uniform'``.
    algorithm
        The algorithm used to compute nearest neighbors. One of ``'auto'``,
        ``'ball_tree'``, ``'kd_tree'``, or ``'brute'``. Defaults to ``'auto'``.
    leaf_size
        Leaf size passed to BallTree or KDTree, affecting query speed and memory.
        Defaults to 30.
    p
        Power parameter for the Minkowski distance metric. ``p=1`` is Manhattan
        distance; ``p=2`` (default) is Euclidean distance.
    radius
        The size of the neighborhood to search when using radius-based model types.
        Applies to ``'radius_regressor'`` and ``'radius_classifier'``. Defaults to 1.0.
    scale_variables
        If True, standardizes predictor variables using StandardScaler before
        modeling. Strongly recommended for KNN because the algorithm is sensitive
        to feature scale. Defaults to True.
    test_size
        The proportion of the dataset used for testing. Defaults to 0.2.
    random_seed
        Controls the randomness of the train-test split for reproducibility.
        Defaults to 412.
    print_model_training_performance
        If True, prints evaluation metrics to the console. For regression: MSE
        and R²; for classification: error rate and classification report.
        Defaults to False.
    data_source_for_plot
        Source citation string displayed in the caption of all generated plots.
        Defaults to None.
    plot_model_test_performance
        Whether to generate a visualization of model performance on the test set
        (scatterplot for regression, confusion matrix heatmap for classification).
        Defaults to True.
    dot_fill_color, line_color, heatmap_color_palette
        Aesthetic settings for the model performance visualization.
    figure_size_for_model_test_performance_plot
        Dimensions (width, height) for the performance plot. Defaults to (8, 6).
    title_for_model_test_performance_plot, subtitle_for_model_test_performance_plot, caption_for_model_test_performance_plot
        Text elements for the performance visualization.
    title_y_indent_for_model_test_performance_plot, subtitle_y_indent_for_model_test_performance_plot, caption_y_indent_for_model_test_performance_plot, x_indent_for_model_test_performance_plot
        Coordinate offsets for text placement in the performance plot.
    plot_feature_importance
        Whether to generate a horizontal bar chart of permutation feature importance
        computed on the test set. Defaults to True.
    top_n_to_highlight
        The number of top features to color differently in the importance chart.
        Defaults to 3.
    highlight_color, fill_transparency
        Aesthetic settings for the feature importance bars.
    figure_size_for_feature_importance_plot
        Dimensions (width, height) for the importance plot. Defaults to (8, 6).
    title_for_feature_importance_plot, subtitle_for_feature_importance_plot, caption_for_feature_importance_plot
        Text elements for the importance visualization.
    title_y_indent_for_feature_importance_plot, subtitle_y_indent_for_feature_importance_plot, caption_y_indent_for_feature_importance_plot
        Coordinate offsets for text placement in the importance plot.
    plot_training_and_test_performance
        Whether to generate a bar chart comparing training vs. test performance.
        Defaults to True.
    training_bar_color
        Hex color for the training bar. Defaults to ``"#3a86ff"``.
    test_bar_color
        Hex color for the test bar. Defaults to ``"#b0170c"``.
    figure_size_for_performance_comparison_plot
        Dimensions (width, height) for the comparison chart. Defaults to (7, 5).
    title_for_performance_comparison_plot, subtitle_for_performance_comparison_plot, caption_for_performance_comparison_plot
        Text elements for the comparison chart. When None, defaults to a
        metric-appropriate string.
    title_y_indent_for_performance_comparison_plot, subtitle_y_indent_for_performance_comparison_plot, caption_y_indent_for_performance_comparison_plot, x_indent_for_performance_comparison_plot
        Coordinate offsets for text placement in the comparison chart.

    Returns
    -------
    estimator or sklearn.pipeline.Pipeline
        The fitted KNN model. If ``scale_variables=True``, a Pipeline containing
        the StandardScaler and the KNN estimator is returned so that new
        observations can be predicted without manual scaling.

    Examples
    --------
    # Predict housing prices using comparable properties
    model = CreateKNearestNeighborsModel(
        df,
        outcome_variable='SalePrice',
        list_of_predictor_variables=['SquareFeet', 'Bedrooms', 'Age']
    )

    # Classify customer churn using a distance-weighted KNN classifier
    model = CreateKNearestNeighborsModel(
        df,
        outcome_variable='Churned',
        list_of_predictor_variables=['Tenure', 'MonthlySpend'],
        model_type='classifier',
        n_neighbors=7,
        weights='distance'
    )

    """
    # Validate model_type
    valid_model_types = ['regressor', 'classifier', 'radius_regressor', 'radius_classifier']
    if model_type not in valid_model_types:
        raise ValueError(
            f"model_type must be one of {valid_model_types}. Got '{model_type}'."
        )

    is_classifier = model_type in ['classifier', 'radius_classifier']

    # Warn about the curse of dimensionality for 5 or more predictors
    if len(list_of_predictor_variables) >= 5:
        print(
            "\nWARNING: You have provided " + str(len(list_of_predictor_variables)) + " predictor "
            "variables. K-Nearest Neighbors works best with 4 or fewer predictors.\n\n"
            "Here is why this matters: KNN makes predictions by finding the data points "
            "that are most similar — or 'nearest' — to a new observation. With just a few "
            "variables, this works well: two customers with similar age and income really "
            "are comparable. But as you add more variables, something counterintuitive "
            "happens: every point in the dataset starts to look roughly the same distance "
            "from every other point. Imagine finding your nearest neighbor first on your "
            "street, then in your city, then your country, then the planet — the more "
            "dimensions you add, the more spread out everything becomes and the harder it "
            "is to find a truly 'close' match. This is called the curse of dimensionality. "
            "Consider reducing the number of predictor variables, or switching to a model "
            "type such as a decision tree or boosted tree that handles many variables better.\n"
        )

    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()

    # Replace inf with nan and drop rows with nan
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.dropna(inplace=True)

    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_seed,
    )

    # Build the KNN estimator based on model_type
    if model_type == 'regressor':
        estimator = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )
    elif model_type == 'classifier':
        estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )
    elif model_type == 'radius_regressor':
        estimator = RadiusNeighborsRegressor(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )
    else:
        estimator = RadiusNeighborsClassifier(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )

    # Wrap in a scaling pipeline if requested
    if scale_variables:
        model = make_pipeline(StandardScaler(), estimator)
    else:
        model = estimator

    # Fit the model
    model.fit(train[list_of_predictor_variables], train[outcome_variable])

    # Add predictions to training and test sets
    train = train.copy()
    test = test.copy()
    train['Predicted'] = model.predict(train[list_of_predictor_variables])
    test['Predicted'] = model.predict(test[list_of_predictor_variables])

    # Compute training and test metrics
    if is_classifier:
        training_metric = 1 - metrics.accuracy_score(train[outcome_variable], train['Predicted'])
        test_metric = 1 - metrics.accuracy_score(test[outcome_variable], test['Predicted'])
    else:
        training_metric = metrics.mean_squared_error(train[outcome_variable], train['Predicted'])
        test_metric = metrics.mean_squared_error(test[outcome_variable], test['Predicted'])

    # Print training performance if requested
    if print_model_training_performance:
        if is_classifier:
            print('Training Error Rate:', training_metric)
            print('Test Error Rate:', test_metric)
            classification_report = metrics.classification_report(test[outcome_variable], test['Predicted'])
            print("Classification Report:\n", classification_report, sep="")
        else:
            print('Training MSE:', training_metric)
            print('Test MSE:', test_metric)
            print('Variance Score:', metrics.r2_score(test[outcome_variable], test['Predicted']))
            print("Note: A variance score of 1 is perfect prediction and 0 means there is no relationship between X and Y.")

    # Plot model performance on the test set
    if plot_model_test_performance:
        plt.figure(figsize=figure_size_for_model_test_performance_plot)

        if is_classifier:
            # Contingency table
            contingency_table = pd.crosstab(
                test['Predicted'],
                test[outcome_variable],
                rownames=['Predicted'],
                colnames=['Actual'],
                margins=True,
                margins_name='Total',
                dropna=False,
            )
            display(contingency_table)

            # Confusion matrix heatmap (column-normalized)
            confusion_matrix = metrics.confusion_matrix(test[outcome_variable], test['Predicted'])
            confusion_matrix = confusion_matrix.transpose()
            confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)

            plt.figure(figsize=(9, 9))
            ax = sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='.0%',
                linewidths=.5,
                square=True,
                cmap=heatmap_color_palette,
            )
            ax.collections[0].colorbar.remove()
        else:
            # Scatterplot of predicted vs. actual with LOWESS smoother
            ax = sns.regplot(
                data=test,
                x=outcome_variable,
                y='Predicted',
                marker='o',
                scatter_kws={
                    'color': dot_fill_color,
                    'alpha': 0.5,
                    'edgecolor': dot_fill_color,
                },
                lowess=True,
                line_kws={'color': line_color},
            )
            # Perfect-prediction reference line
            plt.plot(
                test[outcome_variable],
                test[outcome_variable],
                color='black',
                alpha=0.35,
                linewidth=0.5,
                linestyle='--',
            )

        # Style axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.tick_params(which='major', labelsize=9, color='#666666')

        ax.text(
            x=x_indent_for_model_test_performance_plot,
            y=title_y_indent_for_model_test_performance_plot,
            s=title_for_model_test_performance_plot,
            fontsize=14,
            color="#262626",
            transform=ax.transAxes,
        )
        ax.text(
            x=x_indent_for_model_test_performance_plot,
            y=subtitle_y_indent_for_model_test_performance_plot,
            s=subtitle_for_model_test_performance_plot,
            fontsize=11,
            color="#666666",
            transform=ax.transAxes,
        )

        if not is_classifier:
            ax.yaxis.set_label_coords(-0.1, 0.92)
            ax.yaxis.set_label_text('Predicted', fontsize=10, color="#666666")
            ax.xaxis.set_label_coords(0.9, -0.1)
            ax.xaxis.set_label_text(
                textwrap.fill(outcome_variable, 30, break_long_words=False),
                fontsize=10,
                color="#666666",
                ha='right',
            )

        if caption_for_model_test_performance_plot is not None or data_source_for_plot is not None:
            wrapped_caption = ""
            if caption_for_model_test_performance_plot is not None:
                wrapped_caption = textwrap.fill(caption_for_model_test_performance_plot, 130, break_long_words=False)
            if data_source_for_plot is not None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            ax.text(
                x=x_indent_for_model_test_performance_plot,
                y=caption_y_indent_for_model_test_performance_plot,
                s=wrapped_caption,
                fontsize=8,
                color="#666666",
                transform=ax.transAxes,
            )

        plt.show()

    # Plot feature importance via permutation importance
    if plot_feature_importance:
        perm_result = permutation_importance(
            model,
            test[list_of_predictor_variables],
            test[outcome_variable],
            n_repeats=10,
            random_state=random_seed,
        )

        data_feature_importance = pd.DataFrame({
            'Feature': list_of_predictor_variables,
            'Importance': perm_result.importances_mean,
        })
        data_feature_importance = data_feature_importance.sort_values(by='Importance', ascending=False)
        data_feature_importance['Highlighted'] = np.where(
            data_feature_importance['Feature'].isin(
                data_feature_importance['Feature'].head(top_n_to_highlight)
            ),
            True,
            False,
        )

        plt.figure(figsize=figure_size_for_feature_importance_plot)
        ax = sns.barplot(
            data=data_feature_importance,
            x='Importance',
            y='Feature',
            hue='Highlighted',
            palette={True: highlight_color, False: "#b8b8b8"},
            alpha=fill_transparency,
            dodge=False,
        )
        ax.legend_.remove()

        y_tick_labels = ax.get_yticklabels()
        wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 50)) for label in y_tick_labels]
        ax.set_yticklabels(wrapped_y_tick_labels, fontsize=10, color="#262626")
        ax.get_xaxis().set_ticks([])
        ax.set_xlabel("Permutation Importance", fontsize=10, color="#262626")

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('#b8b8b8')
        ax.spines['bottom'].set_visible(False)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%.3f',
                label_type='edge',
                padding=5,
                fontsize=10,
                color="#262626",
            )

        plt.subplots_adjust(top=0.85)

        longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
        if len(longest_y_tick_label) >= 30:
            x_indent = -0.3
        else:
            x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)

        ax.text(
            x=x_indent,
            y=title_y_indent_for_feature_importance_plot,
            s=title_for_feature_importance_plot,
            fontsize=14,
            color="#262626",
            transform=ax.transAxes,
        )
        ax.text(
            x=x_indent,
            y=subtitle_y_indent_for_feature_importance_plot,
            s=subtitle_for_feature_importance_plot,
            fontsize=11,
            color="#666666",
            transform=ax.transAxes,
        )

        if caption_for_feature_importance_plot is not None or data_source_for_plot is not None:
            wrapped_caption = ""
            if caption_for_feature_importance_plot is not None:
                wrapped_caption = textwrap.fill(caption_for_feature_importance_plot, 110, break_long_words=False)
            if data_source_for_plot is not None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            ax.text(
                x=x_indent,
                y=caption_y_indent_for_feature_importance_plot,
                s=wrapped_caption,
                fontsize=8,
                color="#666666",
                transform=ax.transAxes,
            )

        plt.show()
        plt.clf()

    # Plot training vs. test performance comparison
    if plot_training_and_test_performance:
        if is_classifier:
            chart_title = title_for_performance_comparison_plot or "Training vs. Test Error Rate"
            chart_subtitle = subtitle_for_performance_comparison_plot or "Compares model error rate on the training and test datasets."
            value_fmt = '{:.4f}'
        else:
            chart_title = title_for_performance_comparison_plot or "Training vs. Test MSE"
            chart_subtitle = subtitle_for_performance_comparison_plot or "Compares model error on the training and test datasets."
            value_fmt = '{:,.4f}'

        fig, ax = plt.subplots(figsize=figure_size_for_performance_comparison_plot)
        bar_values = [training_metric, test_metric]
        bar_labels = ['Training', 'Test']
        bar_colors = [training_bar_color, test_bar_color]
        bars = ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.8, width=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                value_fmt.format(height),
                ha='center',
                va='bottom',
                fontsize=10,
                color='#262626',
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_visible(False)
        ax.tick_params(which='major', labelsize=9, color='#666666')
        ax.yaxis.set_ticks([])
        plt.subplots_adjust(top=0.85)

        ax.text(
            x=x_indent_for_performance_comparison_plot,
            y=title_y_indent_for_performance_comparison_plot,
            s=chart_title,
            fontsize=14,
            color='#262626',
            transform=ax.transAxes,
        )
        ax.text(
            x=x_indent_for_performance_comparison_plot,
            y=subtitle_y_indent_for_performance_comparison_plot,
            s=chart_subtitle,
            fontsize=11,
            color='#666666',
            transform=ax.transAxes,
        )

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
                transform=ax.transAxes,
            )

        plt.show()
        plt.clf()

    return model
