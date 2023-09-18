# Load pacakges
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Function to create linear regression model
def CreateLinearRegressionModel(dataframe,
                                outcome_variable,
                                list_of_predictor_variables,
                                # Model parameters
                                scale_predictor_variables=True,
                                test_size=0.2,
                                max_iterations=1000,
                                learning_rate=0.01,
                                lambda_for_regularization=0.001,
                                random_seed=412,
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
                                caption_y_indent_for_feature_importance_plot=-0.15):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Keep complete cases
    dataframe.dropna(inplace = True)
    dataframe = dataframe[np.isfinite(dataframe).all(1)]
    print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
    # Scale the predictors, if requested
    if scale_predictor_variables:
        # Scale predictors
        scaler = StandardScaler()
        dataframe[list_of_predictor_variables] = scaler.fit_transform(dataframe[list_of_predictor_variables])
        
    # Show the peak-to-peak range of each predictor
    print("\nPeak-to-peak range of each predictor:")
    print(np.ptp(dataframe[list_of_predictor_variables], axis=0))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create linear regression object
    model = linear_model.SGDRegressor(
        loss='squared_error',
        max_iter=max_iterations,
        eta0=learning_rate,
        alpha=lambda_for_regularization,
        random_state=random_seed,
        fit_intercept=True,
    )
    
    # Train the model using the training sets and show fitting summary
    model.fit(train[list_of_predictor_variables], train[outcome_variable])
    print(f"\nNumber of iterations completed: {model.n_iter_}")
    print(f"Number of weight updates: {model.t_}")
    
    # # Show parameters of the model
    # b_norm = model.intercept_
    # w_norm = model.coef_
    # print(f"\nModel parameters:    w: {w_norm}, b:{b_norm}")
    
    # Add predictions to test set
    test['Predicted'] = model.predict(test[list_of_predictor_variables])
    
    # Show mean squared error if outcome is numerical
    print('Mean Squared Error:', metrics.mean_squared_error(test[outcome_variable], test['Predicted']))
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
            fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent_for_model_test_performance_plot,
            y=subtitle_y_indent_for_model_test_performance_plot,
            s=subtitle_for_model_test_performance_plot,
            fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
        
        # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
        ax.yaxis.set_label_coords(-0.1, 0.92)
        ax.yaxis.set_label_text(
            'Predicted',
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
        # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(
            outcome_variable,
            fontname="Arial",
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
                fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
            
            # Show the plot
            plt.show()
    
    # Plot feature importance if requested
    if plot_feature_importance:
        # Get importance
        importance = model.coef_
        
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
                'Importance': model.coef_
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
        ax.set_xlabel("Beta Coefficent", fontsize=10, fontname="Arial", color="#262626")
        
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
                y=caption_y_indent,
                s=wrapped_caption,
                fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
            
        # Show the plot
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


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# linear_reg_model = CreateLinearRegressionModel(dataframe=iris,
#                                                outcome_variable='sepal length (cm)',
#                                                list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#                                                scale_predictor_variables=False)

