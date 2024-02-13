# Load packages
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
import textwrap

# Declare function
def CreateDecisionTreeModel(dataframe,
                            outcome_variable,
                            list_of_predictor_variables,
                            is_outcome_categorical=True,
                            # Output arguments
                            print_model_training_performance=False,
                            # Model training arguments
                            test_size=0.2,
                            categorical_splitting_criterion='entropy',
                            numerical_splitting_criterion='mse',
                            maximum_depth=None,
                            minimum_impurity_decrease=0.0,
                            random_seed=412,
                            filter_nulls=False,
                            # All plot arguments
                            data_source_for_plot=None,
                            # Model performance plot arguments
                            plot_model_test_performance=True,
                            dot_fill_color="#999999",
                            fitted_line_type=None,
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
                            subtitle_for_feature_importance_plot="Shows the predictive power of each feature in the model.",
                            caption_for_feature_importance_plot=None,
                            title_y_indent_for_feature_importance_plot=1.15,
                            subtitle_y_indent_for_feature_importance_plot=1.1,
                            caption_y_indent_for_feature_importance_plot=-0.15,
                            # Decision tree plot arguments
                            plot_decision_tree=False,
                            decision_tree_plot_size=(20, 20),
                            print_decision_rules=False):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Drop rows with infinite values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with missing values if filter_nulls is True
    if filter_nulls:
        dataframe = dataframe.dropna()
    # print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
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
    if print_model_training_performance:
        if is_outcome_categorical == False:
            print('Mean Squared Error:', metrics.mean_squared_error(test[outcome_variable], test['Predicted']))
            print('Variance Score:', metrics.r2_score(test[outcome_variable], test['Predicted']))
            print("Note: A variance score of 1 is perfect prediction and 0 means that there is no linear relationship between X and Y.")
        # Show accuracy if outcome is categorical
        else:
            classifcation_report = metrics.classification_report(test[outcome_variable], test['Predicted'])
            print("Classification Report:\n", classifcation_report, sep="")
    
    # Print decision rules if requested
    if print_decision_rules:
        r = export_text(
            model,
            feature_names=list_of_predictor_variables
        )
        print(r)
    
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
        # Set the size of the plot
        plt.figure(figsize=figure_size_for_model_test_performance_plot)
        
        # If outcome is categorical, create a heatmap
        if is_outcome_categorical:
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
            
            # Display as a markdown table
            display(contingency_table)
            
            # Create a confusion matrix
            confusion_matrix = metrics.confusion_matrix(
                test[outcome_variable], 
                test['Predicted']
            )
            
            # Transpose the confusion matrix
            confusion_matrix = confusion_matrix.transpose()
            
            # Convert the confusion matrix to a percentage using the column sums
            confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)
            
            # Generate a heatmap of the confusion matrix
            plt.figure(figsize=(9,9))
            ax = sns.heatmap(
                confusion_matrix, 
                annot=True, 
                # Format numbers as percentages
                fmt='.0%', 
                linewidths=.5, 
                square=True, 
                cmap=heatmap_color_palette,
            )
            
            # Remove the color bar
            ax.collections[0].colorbar.remove()

        # Plot the residuals if outcome is numerical
        else:
            # Generate a scatterplot of the predicted vs. observed outcome
            ax = sns.regplot(
                data=test,
                x=outcome_variable,
                y='Predicted',
                marker='o',
                scatter_kws={
                    'color': dot_fill_color,
                    'alpha': 0.5,
                    'linewidth': 0.5,
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
        ax.yaxis.set_label_coords(-0.1, 0.92)
        ax.yaxis.set_label_text(
            'Predicted',
            # fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(
            outcome_variable,
            # fontname="Arial",
            fontsize=10,
            color="#666666"
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
            "Importance", 
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
     
    # Return the model
    return model

