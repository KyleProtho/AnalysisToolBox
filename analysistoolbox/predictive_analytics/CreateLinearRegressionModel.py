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
                                caption_y_indent_for_feature_importance_plot=-0.15):
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
    
    # Add predictions to test set
    test['Predicted'] = model.predict(test[list_of_predictor_variables])
    
    # Show mean squared error if outcome is numerical
    if print_model_training_performance:
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
    
    # Return the model
    return model

