import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

def CreateDecisionTreeModel(dataframe,
                            outcome_variable,
                            list_of_predictor_variables,
                            is_outcome_categorical=True,
                            # Model training arguments
                            test_size=0.2,
                            categorical_splitting_criterion='entropy',
                            numerical_splitting_criterion='mse',
                            maximum_depth=None,
                            minimum_impurity_decrease=0.0,
                            random_seed=412,
                            # Model performance plot arguments
                            plot_model_test_performance=True,
                            # Feature importance plot arguments
                            plot_feature_importance=True,
                            top_n_to_highlight=3,
                            highlight_color="#b0170c",
                            fill_transparency=0.8,
                            figure_size_for_feature_importance_plot=(8, 6),
                            title_for_feature_importance_plot="Feauture Importance",
                            subtitle_for_feature_importance_plot="Shows the predictive power of each feature in the model.",
                            caption__feature_importance_plot=None,
                            title_y_indent_for_feature_importance_plot=1.15,
                            subtitle_y_indent_for_feature_importance_plot=1.1,
                            caption_y_indent_for_feature_importance_plot=-0.15,
                            # Decision tree plot arguments
                            plot_decision_tree=True,
                            decision_tree_plot_size=(20, 20),
                            print_decision_rules=False):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Drop rows with infinite values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with missing values
    dataframe = dataframe.dropna()
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create decision tree model
    if is_outcome_categorical:
        model = DecisionTreeClassifier(
            criterion=categorical_splitting_criterion,
            max_depth=maximum_depth,
            min_impurity_decrease=minimum_impurity_decrease,
            random_state=random_seed
        )
    else:
        model = DecisionTreeRegressor(
            criterion=numerical_splitting_criterion,
            max_depth=maximum_depth,
            min_impurity_decrease=minimum_impurity_decrease,
            random_state=random_seed
        )
        
    # Fit the model
    model = model.fit(train[list_of_predictor_variables], train[outcome_variable])
    
    # Add predictions to test set
    test['Predicted'] = model.predict(test[list_of_predictor_variables])
    
    # Show mean squared error and variance if outcome is numerical
    if is_outcome_categorical == False:
        print('Mean Squared Error:', metrics.mean_squared_error(test[outcome_variable], test['Predicted']))
        print('Variance Score:', metrics.r2_score(test[outcome_variable], test['Predicted']))
        print("Note: A variance score of 1 is perfect prediction and 0 means that there is no linear relationship between X and Y.")
    
    # Print decision rules if requested
    if print_decision_rules:
        r = export_text(
            model,
            feature_names=list_of_predictor_variables
        )
        print(r)
        
    # Plot feature importance if requested
    if plot_feature_importance:
        # Create dataframe of feature importance
        data_feauture_importance = pd.DataFrame(
            data={
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
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
        ax.set_yticklabels(wrapped_y_tick_labels, fontsize=10, fontname="Arial", color="#262626")
        
        # Remove a-axis tick labels
        ax.get_xaxis().set_ticks([])
        
        # Format x-axis label
        ax.set_xlabel("Importance", fontsize=10, fontname="Arial", color="#262626")
        
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
                fontname="Arial", 
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
            fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent_for_feature_importance_plot,
            s=subtitle_for_feature_importance_plot,
            fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
    
    # Plot decision tree if requested
    if plot_decision_tree:
        plt.figure(figsize=decision_tree_plot_size)
        plot_tree(
            model, 
            feature_names=list_of_predictor_variables,
            filled=True,
            rounded=True,
            precision=3
        )
    
    # Print the confusion matrix if outcome is categorical
    if plot_model_test_performance:
        if is_outcome_categorical:
            # Calculate the accuracy score if outcome is categorical
            score = model.score(test[list_of_predictor_variables], test[outcome_variable])
            confusion_matrix = metrics.confusion_matrix(
                test[outcome_variable], 
                test['Predicted']
            )
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
        # Plot the residuals if outcome is numerical
        else:
            plt.figure(figsize=(9, 9))
            sns.regplot(
                    data=test,
                    x=outcome_variable,
                    y='Predicted',
                )
            plt.plot(test[outcome_variable], test[outcome_variable], color='black', alpha=0.35)
            plt.title('Predicted vs. Observed Outcome', size = 15)
            plt.show()
        
    # Return the model
    return model

# Test the function
from sklearn import datasets
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
iris['species'] = datasets.load_iris(as_frame=True).target
# CATEGORICAL OUTCOME
species_desc_tree_model = CreateDecisionTreeModel(
    dataframe=iris,
    outcome_variable='species',
    list_of_predictor_variables=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
)
# # # NUMERICAL OUTCOME
# # sep_len_desc_tree_model = CreateDecisionTreeModel(
# #     dataframe=iris,
# #     outcome_variable='sepal length (cm)',
# #     is_outcome_categorical=False,
# #     list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
# #     maximum_depth=5
# # )
