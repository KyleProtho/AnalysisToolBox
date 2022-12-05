import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# Function that creates a boosted tree model
def CreateBoostedTreeModel(dataframe,
                           outcome_variable,
                           list_of_predictor_variables,
                           maximum_depth=None,
                           is_outcome_categorical=True,
                           plot_model_test_performance=True,
                           test_size=0.2,
                           random_seed=412):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create boosted tree model
    if is_outcome_categorical:
        model = XGBClassifier(
            max_depth=maximum_depth,
            random_state=random_seed
        )
    else:
        model = XGBRegressor(
            max_depth=maximum_depth,
            random_state=random_seed
        )
    
    # Fit the model
    model = model.fit(train[list_of_predictor_variables], train[outcome_variable])
    
    # Add predictions to test set
    test['Predicted'] = model.predict(test[list_of_predictor_variables])
     
    # Print the confusion matrix if outcome is categorical
    if plot_model_test_performance:
        if is_outcome_categorical:
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
# # CATEGORICAL OUTCOME
# species_boosted_tree_model = CreateBoostedTreeModel(
#     dataframe=iris,
#     outcome_variable='species',
#     list_of_predictor_variables=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#     # maximum_depth=5
# )
# NUMERICAL OUTCOME
sep_len_boosted_tree_model = CreateBoostedTreeModel(
    dataframe=iris,
    outcome_variable='sepal length (cm)',
    is_outcome_categorical=False,
    list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    maximum_depth=5
)
