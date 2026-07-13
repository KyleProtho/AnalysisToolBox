# Load packages
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import textwrap

# Declare function
def CreateBayesClassifier(dataframe,
                          outcome_variable,
                          list_of_predictor_variables,
                          # Model parameters
                          naive_bayes_type='gaussian',
                          alpha=1.0,
                          var_smoothing=1e-9,
                          # Model training arguments
                          test_size=0.2,
                          random_seed=412,
                          filter_nulls=False,
                          # Output arguments
                          print_model_training_performance=False,
                          # All plot arguments
                          data_source_for_plot=None,
                          # Model performance plot arguments
                          plot_model_test_performance=True,
                          heatmap_color_palette="Blues",
                          figure_size_for_model_test_performance_plot=(8, 6),
                          title_for_model_test_performance_plot="Model Performance",
                          subtitle_for_model_test_performance_plot="The predicted values vs. the actual values in the test dataset.",
                          caption_for_model_test_performance_plot=None,
                          title_y_indent_for_model_test_performance_plot=1.09,
                          subtitle_y_indent_for_model_test_performance_plot=1.05,
                          caption_y_indent_for_model_test_performance_plot=-0.215,
                          x_indent_for_model_test_performance_plot=-0.115,
                          # Feature discriminability plot arguments
                          plot_feature_importance=True,
                          top_n_to_highlight=3,
                          highlight_color="#b0170c",
                          fill_transparency=0.8,
                          figure_size_for_feature_importance_plot=(8, 6),
                          title_for_feature_importance_plot="Feature Discriminability",
                          subtitle_for_feature_importance_plot="Shows how useful each feature is for distinguishing between classes.",
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
    Train, evaluate, and visualize a Naive Bayes classification model.

    This function supports five scikit-learn Naive Bayes variants and handles
    the full workflow: data preparation, train/test splitting, model fitting,
    performance metrics, and three diagnostic visualizations (confusion matrix,
    feature discriminability, and training vs. test error rate comparison).

    Naive Bayes classifiers are fast, interpretable probabilistic models that
    apply Bayes' theorem with a strong (naïve) assumption of conditional
    independence between features given the class label. Despite this
    simplification, they often perform surprisingly well in practice,
    especially when:

    * Training data are limited and simpler models generalize better
    * Features are genuinely or approximately independent (e.g., word counts
      in text classification after tokenization)
    * A fast, interpretable baseline is needed before trying complex models
    * Class-conditional distributions are well-separated in feature space
    * Real-time prediction is required, since scoring is O(n_features)

    The five supported variants reflect different assumptions about the
    feature distributions:

    * ``'gaussian'`` — continuous features, estimates class-conditional normal
      distributions for each predictor. Best general-purpose choice.
    * ``'multinomial'`` — non-negative integer count features (e.g., word
      frequency counts). Common in text classification.
    * ``'bernoulli'`` — binary features (e.g., word presence/absence). Also
      common in text classification; penalizes absence of features.
    * ``'complement'`` — an adaptation of MultinomialNB for imbalanced
      datasets; trains on the complement of each class.
    * ``'categorical'`` — explicitly categorical (ordinal-encoded) features.
      Requires non-negative integer feature values.

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing training and testing data.
    outcome_variable
        The name of the target column to be predicted (must be categorical).
    list_of_predictor_variables
        A list of column names used as features for the model.
    naive_bayes_type
        Which Naive Bayes variant to use. One of ``'gaussian'``,
        ``'multinomial'``, ``'bernoulli'``, ``'complement'``, or
        ``'categorical'``. Defaults to ``'gaussian'``.
    alpha
        Laplace/Lidstone smoothing parameter for ``'multinomial'``,
        ``'bernoulli'``, ``'complement'``, and ``'categorical'`` variants.
        Has no effect for ``'gaussian'``. Defaults to 1.0.
    var_smoothing
        Portion of the largest variance across all features added to variances
        for numerical stability in ``'gaussian'`` mode. Has no effect for other
        variants. Defaults to 1e-9.
    test_size
        The proportion of the dataset reserved for testing. Defaults to 0.2.
    random_seed
        Random state for reproducible train/test splits. Defaults to 412.
    filter_nulls
        If True, drops all rows containing any NaN values across the selected
        columns before splitting. Defaults to False.
    print_model_training_performance
        If True, prints Training Error Rate, Test Error Rate, and a full
        Classification Report to the console. Defaults to False.
    data_source_for_plot
        Optional source citation appended to every plot caption. Defaults to None.
    plot_model_test_performance
        If True, renders a normalized confusion matrix heatmap on the test set.
        Defaults to True.
    heatmap_color_palette
        A seaborn/matplotlib colormap name for the confusion matrix.
        Defaults to ``"Blues"``.
    figure_size_for_model_test_performance_plot
        Dimensions (width, height) for the confusion matrix plot. Defaults to (8, 6).
    title_for_model_test_performance_plot
        Title text for the confusion matrix plot.
    subtitle_for_model_test_performance_plot
        Subtitle text for the confusion matrix plot.
    caption_for_model_test_performance_plot
        Optional caption text for the confusion matrix plot.
    title_y_indent_for_model_test_performance_plot
        Vertical position of the title in axes-fraction coordinates. Defaults to 1.09.
    subtitle_y_indent_for_model_test_performance_plot
        Vertical position of the subtitle in axes-fraction coordinates. Defaults to 1.05.
    caption_y_indent_for_model_test_performance_plot
        Vertical position of the caption in axes-fraction coordinates. Defaults to -0.215.
    x_indent_for_model_test_performance_plot
        Horizontal starting position for the confusion matrix plot text. Defaults to -0.115.
    plot_feature_importance
        If True, renders a horizontal bar chart of feature discriminability scores.
        For ``'gaussian'``, this is the standard deviation of class-conditional means
        across classes. For all other variants, this is the peak-to-peak range of
        log-probabilities across classes. Defaults to True.
    top_n_to_highlight
        Number of top features to emphasize in ``highlight_color``. Defaults to 3.
    highlight_color
        Hex color for the top-N feature bars. Defaults to ``"#b0170c"``.
    fill_transparency
        Alpha transparency for all feature bars. Defaults to 0.8.
    figure_size_for_feature_importance_plot
        Dimensions (width, height) for the discriminability plot. Defaults to (8, 6).
    title_for_feature_importance_plot
        Title text for the discriminability plot.
    subtitle_for_feature_importance_plot
        Subtitle text for the discriminability plot.
    caption_for_feature_importance_plot
        Optional caption text for the discriminability plot.
    title_y_indent_for_feature_importance_plot
        Vertical position of the title. Defaults to 1.15.
    subtitle_y_indent_for_feature_importance_plot
        Vertical position of the subtitle. Defaults to 1.1.
    caption_y_indent_for_feature_importance_plot
        Vertical position of the caption. Defaults to -0.15.
    plot_training_and_test_performance
        If True, renders a bar chart comparing training vs. test error rates.
        Defaults to True.
    training_bar_color
        Hex color for the training bar. Defaults to ``"#3a86ff"``.
    test_bar_color
        Hex color for the test bar. Defaults to ``"#b0170c"``.
    figure_size_for_performance_comparison_plot
        Dimensions (width, height) for the comparison chart. Defaults to (7, 5).
    title_for_performance_comparison_plot
        Title text for the comparison chart. Defaults to ``"Training vs. Test Error Rate"``.
    subtitle_for_performance_comparison_plot
        Subtitle text for the comparison chart.
    caption_for_performance_comparison_plot
        Optional caption text for the comparison chart.
    title_y_indent_for_performance_comparison_plot
        Vertical position of the title. Defaults to 1.10.
    subtitle_y_indent_for_performance_comparison_plot
        Vertical position of the subtitle. Defaults to 1.05.
    caption_y_indent_for_performance_comparison_plot
        Vertical position of the caption. Defaults to -0.15.
    x_indent_for_performance_comparison_plot
        Horizontal starting position for comparison chart text. Defaults to -0.115.

    Returns
    -------
    sklearn.naive_bayes.GaussianNB or similar
        The fitted Naive Bayes model object.

    Examples
    --------
    # Classify iris species using GaussianNB (continuous features)
    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    model = CreateBayesClassifier(
        df,
        outcome_variable='species',
        list_of_predictor_variables=iris.feature_names.tolist()
    )

    # Classify text documents using MultinomialNB (word count features)
    model = CreateBayesClassifier(
        df_counts,
        outcome_variable='category',
        list_of_predictor_variables=vocabulary_columns,
        naive_bayes_type='multinomial',
        alpha=0.5
    )

    Teaching Note
    -------------
    The "naïve" in Naive Bayes refers to the assumption that all predictor
    variables are conditionally independent given the class label. In other
    words, knowing the value of one feature tells you nothing additional about
    any other feature, once you know the class. This assumption almost never
    holds exactly in practice — yet Naive Bayes classifiers are remarkably
    robust to its violation.

    The key insight is that even when features are correlated, the model can
    still rank classes correctly (produce the right argmax prediction) as long
    as the conditional probability estimates are in the right relative order.
    Errors in the probability magnitudes cancel out across features more often
    than they accumulate.

    Naive Bayes is a useful first model to reach for because:

    1. **Speed**: Training is a single pass over the data to compute class
       priors and per-feature conditional statistics. Scoring is O(n_features).
    2. **Data efficiency**: It performs well with small datasets where complex
       models would overfit.
    3. **Interpretability**: The class-conditional means (GaussianNB) or
       log-probability tables (MultinomialNB) are directly inspectable.
    4. **Calibrated probabilities**: When assumptions are met, the posterior
       probabilities are well-calibrated, which is useful for decision-making
       under uncertainty.
    5. **Baseline value**: A Naive Bayes model that outperforms a more complex
       one is a strong signal to investigate whether feature engineering can
       replace model complexity.

    """
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()

    # Drop rows with infinite values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing values if filter_nulls is True
    if filter_nulls:
        dataframe = dataframe.dropna()

    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_seed
    )

    # Select and instantiate the Naive Bayes variant
    naive_bayes_type = naive_bayes_type.lower().strip()
    if naive_bayes_type == 'gaussian':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(var_smoothing=var_smoothing)
    elif naive_bayes_type == 'multinomial':
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=alpha)
    elif naive_bayes_type == 'bernoulli':
        from sklearn.naive_bayes import BernoulliNB
        model = BernoulliNB(alpha=alpha)
    elif naive_bayes_type == 'complement':
        from sklearn.naive_bayes import ComplementNB
        model = ComplementNB(alpha=alpha)
    elif naive_bayes_type == 'categorical':
        from sklearn.naive_bayes import CategoricalNB
        model = CategoricalNB(alpha=alpha)
    else:
        raise ValueError(
            f"naive_bayes_type must be one of 'gaussian', 'multinomial', "
            f"'bernoulli', 'complement', or 'categorical'. Got: '{naive_bayes_type}'"
        )

    # Fit the model
    model = model.fit(train[list_of_predictor_variables], train[outcome_variable])

    # Add predictions to training and test sets
    train = train.copy()
    test = test.copy()
    train['Predicted'] = model.predict(train[list_of_predictor_variables])
    test['Predicted'] = model.predict(test[list_of_predictor_variables])

    # Compute training and test error rates
    training_metric = 1 - metrics.accuracy_score(train[outcome_variable], train['Predicted'])
    test_metric = 1 - metrics.accuracy_score(test[outcome_variable], test['Predicted'])

    # Print training and test performance
    if print_model_training_performance:
        print('Training Error Rate:', training_metric)
        print('Test Error Rate:', test_metric)
        classification_report = metrics.classification_report(test[outcome_variable], test['Predicted'])
        print("Classification Report:\n", classification_report, sep="")

    # Plot confusion matrix on the test set
    if plot_model_test_performance:
        # Generate a contingency table using pandas
        contingency_table = pd.crosstab(
            test['Predicted'],
            test[outcome_variable],
            rownames=['Predicted'],
            colnames=['Actual'],
            margins=True,
            margins_name='Total',
            dropna=False
        )
        display(contingency_table)

        # Create a confusion matrix
        confusion_matrix = metrics.confusion_matrix(
            test[outcome_variable],
            test['Predicted']
        )

        # Transpose so rows are predicted, columns are actual
        confusion_matrix = confusion_matrix.transpose()

        # Normalize each column to percentages
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)

        # Generate a heatmap of the confusion matrix
        plt.figure(figsize=(9, 9))
        ax = sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='.0%',
            linewidths=.5,
            square=True,
            cmap=heatmap_color_palette,
        )

        # Remove the color bar
        ax.collections[0].colorbar.remove()

        # Remove top and right spines, and set bottom and left spines to gray
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')

        # Format tick labels
        ax.tick_params(which='major', labelsize=9, color='#666666')

        # Set the title
        ax.text(
            x=x_indent_for_model_test_performance_plot,
            y=title_y_indent_for_model_test_performance_plot,
            s=title_for_model_test_performance_plot,
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )

        # Set the subtitle
        ax.text(
            x=x_indent_for_model_test_performance_plot,
            y=subtitle_y_indent_for_model_test_performance_plot,
            s=subtitle_for_model_test_performance_plot,
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )

        # Move the y-axis label to the top
        ax.yaxis.set_label_coords(-0.1, 0.92)
        ax.yaxis.set_label_text('Predicted', fontsize=10, color="#666666")

        # Move the x-axis label to the right
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(outcome_variable, fontsize=10, color="#666666")

        # Add a word-wrapped caption if one is provided
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
                transform=ax.transAxes
            )

        plt.show()

    # Plot feature discriminability
    if plot_feature_importance:
        # Compute a discriminability score per feature based on the NB variant
        if naive_bayes_type == 'gaussian':
            # Standard deviation of class-conditional means across classes
            discriminability_scores = np.std(model.theta_, axis=0)
        elif naive_bayes_type == 'categorical':
            # Average inter-class range of log-probabilities across category levels
            discriminability_scores = np.array([
                np.mean(np.max(fp, axis=0) - np.min(fp, axis=0))
                for fp in model.feature_log_prob_
            ])
        else:
            # Peak-to-peak range of feature log-probabilities across classes
            discriminability_scores = (
                np.max(model.feature_log_prob_, axis=0)
                - np.min(model.feature_log_prob_, axis=0)
            )

        # Build a sorted DataFrame of features and their discriminability scores
        data_feature_discriminability = pd.DataFrame(
            data={
                'Feature': list_of_predictor_variables,
                'Discriminability': discriminability_scores
            }
        )
        data_feature_discriminability = data_feature_discriminability.sort_values(
            by='Discriminability', ascending=False
        )

        # Flag the top-N features for highlighting
        data_feature_discriminability['Highlighted'] = np.where(
            data_feature_discriminability['Feature'].isin(
                data_feature_discriminability['Feature'].head(top_n_to_highlight)
            ),
            True,
            False
        )

        # Plot horizontal bar chart
        plt.figure(figsize=figure_size_for_feature_importance_plot)
        ax = sns.barplot(
            data=data_feature_discriminability,
            x='Discriminability',
            y='Feature',
            hue='Highlighted',
            palette={True: highlight_color, False: "#b8b8b8"},
            alpha=fill_transparency,
            dodge=False
        )

        # Remove the legend
        ax.legend_.remove()

        # Format and wrap y-axis tick labels
        y_tick_labels = ax.get_yticklabels()
        wrapped_y_tick_labels = ['\n'.join(textwrap.wrap(label.get_text(), 50)) for label in y_tick_labels]
        ax.set_yticklabels(wrapped_y_tick_labels, fontsize=10, color="#262626")

        # Remove x-axis tick marks
        ax.get_xaxis().set_ticks([])

        # Format x-axis label
        ax.set_xlabel("Discriminability Score", fontsize=10, color="#262626")

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('#b8b8b8')
        ax.spines['bottom'].set_visible(False)

        # Add data labels at the end of each bar
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%.4f',
                label_type='edge',
                padding=5,
                fontsize=10,
                color="#262626"
            )

        # Add space between the title and the plot
        plt.subplots_adjust(top=0.85)

        # Determine x indent based on longest y tick label
        longest_y_tick_label = max(wrapped_y_tick_labels, key=len)
        if len(longest_y_tick_label) >= 30:
            x_indent = -0.3
        else:
            x_indent = -0.005 - (len(longest_y_tick_label) * 0.011)

        # Set the title
        ax.text(
            x=x_indent,
            y=title_y_indent_for_feature_importance_plot,
            s=title_for_feature_importance_plot,
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )

        # Set the subtitle
        ax.text(
            x=x_indent,
            y=subtitle_y_indent_for_feature_importance_plot,
            s=subtitle_for_feature_importance_plot,
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )

        # Add a word-wrapped caption if one is provided
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
                transform=ax.transAxes
            )

        plt.show()
        plt.clf()

    # Plot training vs. test error rate comparison
    if plot_training_and_test_performance:
        chart_title = title_for_performance_comparison_plot or "Training vs. Test Error Rate"
        chart_subtitle = subtitle_for_performance_comparison_plot or "Compares model error rate on the training and test datasets."

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
                '{:.4f}'.format(height),
                ha='center', va='bottom', fontsize=10, color='#262626'
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
            fontsize=14, color="#262626", transform=ax.transAxes
        )
        ax.text(
            x=x_indent_for_performance_comparison_plot,
            y=subtitle_y_indent_for_performance_comparison_plot,
            s=chart_subtitle,
            fontsize=11, color="#666666", transform=ax.transAxes
        )
        if caption_for_performance_comparison_plot is not None:
            ax.text(
                x=x_indent_for_performance_comparison_plot,
                y=caption_y_indent_for_performance_comparison_plot,
                s=textwrap.fill(caption_for_performance_comparison_plot, 80, break_long_words=False),
                fontsize=8, color="#666666", transform=ax.transAxes
            )
        plt.show()
        plt.clf()

    # Return the fitted model
    return model
