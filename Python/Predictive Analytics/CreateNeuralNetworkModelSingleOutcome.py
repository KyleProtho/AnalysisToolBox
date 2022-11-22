import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function that creates a neural network model using TensorFlow
def CreateNeuralNetwork_SingleOutcome(dataframe,
                                      outcome_variable,
                                      list_of_predictor_variables,
                                      number_of_hidden_layers,
                                      is_outcome_categorical=True,
                                      test_size=0.2,
                                      scale_predictor_variables=True,
                                      plot_loss=True,
                                      plot_model_test_performance=True,
                                      show_predictor_ranges=True,
                                      initial_learning_rate=0.01,
                                      number_of_steps_gradient_descent=100,
                                      lambda_for_regularization=0.001,
                                      random_seed=412):
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Set the random seed
    tf.random.set_seed(random_seed)
    
    # Use Keras to create a normalization layer
    if scale_predictor_variables:
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(dataframe[list_of_predictor_variables]))  # Learns the statistics of the data
        norm_predictor_variables = normalizer(np.array(dataframe[list_of_predictor_variables]))
        dataframe[list_of_predictor_variables] = pd.DataFrame(norm_predictor_variables.numpy(), columns=list_of_predictor_variables)

    # Show the peak-to-peak range of each predictor
    if show_predictor_ranges:
        print("\nPeak-to-peak range of each predictor:")
        print(np.ptp(dataframe[list_of_predictor_variables], axis=0))
        
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Choose activation function
    if is_outcome_categorical: 
        if len(dataframe[outcome_variable].unique()) == 2:
            activation_function = 'sigmoid'
        else:
            activation_function = 'linear'
    else:
        if dataframe[outcome_variable].min() >= 0:
            activation_function = 'relu'
        else:
            activation_function = 'linear'
                
    # Create dictionary of layers to be used in the neural network
    dict_layers = {}
    
    # Create input layer
    dict_layers['Input layer'] = tf.keras.layers.Input(
        shape=(len(list_of_predictor_variables),), 
        name='input_layer'
    )
    
    # Create hidden layers
    for i in range(number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        if i < 10:
            layer_name = 'layer_0' + str(i + 1)
        else:
            layer_name = 'layer_' + str(i + 1)
        dict_layers[key_text] = tf.keras.layers.Dense(
            10 + ((number_of_hidden_layers - 1 - i) * 10), 
            activation='relu', 
            name=layer_name,
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
        
    # Create output layer
    if is_outcome_categorical and len(dataframe[outcome_variable].unique()) > 2:
        dict_layers['Output layer'] = tf.keras.layers.Dense(
            len(dataframe[outcome_variable].unique()), 
            activation=activation_function, 
            name='softmax_layer',
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
    else:
        dict_layers['Output layer'] = tf.keras.layers.Dense(
            1, 
            activation=activation_function, 
            name='final_layer',
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
    
    # Create list of layers to be used in the neural network
    list_of_layers = [dict_layers['Input layer']]
    for i in range(number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        list_of_layers.append(dict_layers[key_text])
    list_of_layers.append(dict_layers['Output layer'])
    
    # Create sequential model
    model = tf.keras.Sequential(list_of_layers) 
    
    # Show model summary
    model.summary()
    
    # Define loss function and optimizer
    if is_outcome_categorical:
        if len(dataframe[outcome_variable].unique()) == 2:
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                metrics=['accuracy']
            )
        else:
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                metrics=['accuracy']
            )
    else:
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            metrics=['mean_squared_error']
        )
    
    # Train the model
    loss_history = model.fit(
        train[list_of_predictor_variables].values,
        train[outcome_variable].values,
        epochs=number_of_steps_gradient_descent,
        verbose=0
    )
    if plot_loss:
        plt.figure(figsize=(9,9))
        sns.lineplot(
            x=range(1, number_of_steps_gradient_descent + 1),
            y=loss_history.history['loss']
        )
        plt.title('Loss Curve', size = 15)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    # Test the model
    predictions = model.predict(test[list_of_predictor_variables].values)
    if is_outcome_categorical and len(dataframe[outcome_variable].unique()) > 2:
        predictions = tf.nn.softmax(predictions).numpy()  # Convert to probabilities
        predictions = pd.DataFrame(predictions).idxmax(axis=1).values  # Convert to class labels
    test['Predicted'] = predictions
    
    # Plot results of model test
    if plot_model_test_performance:
        plt.figure(figsize=(9,9))
        if is_outcome_categorical:
            if len(dataframe[outcome_variable].unique()) == 2:
                sns.scatterplot(
                    data=test,
                    x=outcome_variable,
                    y='Predicted'
                )
                plt.title('Probability vs. Observed Outcome', size = 15)
                plt.ylabel('Predicted probability')
                plt.ylim(0, 1.05)
            else:
                confusion_matrix = metrics.confusion_matrix(
                    test[outcome_variable], 
                    test['Predicted']
                )
                sns.heatmap(
                    confusion_matrix, 
                    annot=True, 
                    fmt=".3f", 
                    linewidths=.5, 
                    square=True, 
                    cmap='Blues_r'
                )
                plt.title('Confusion Matrix', size = 15)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
        else:
            sns.regplot(
                data=test,
                x=outcome_variable,
                y='Predicted'
            )
            plt.plot(test[outcome_variable], test[outcome_variable], color='black', alpha=0.35)
            plt.title('Predicted vs. Observed Outcome', size = 15)
        plt.show()
    
    # Return the model
    return(model)


# # Test the function
# from sklearn.datasets import load_iris
# iris = pd.DataFrame(load_iris(as_frame=True).data)
# iris['species'] = load_iris(as_frame=True).target
# # # CATEGORICAL OUTCOME
# # # iris = iris[iris['species'] != 2]
# # species_neural_net_model = CreateNeuralNetwork_SingleOutcome(
# #     dataframe=iris,
# #     outcome_variable='species',
# #     is_outcome_categorical=True,
# #     list_of_predictor_variables=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
# #     number_of_hidden_layers=2,
# #     scale_predictor_variables=True
# # )
# # NUMERICAL OUTCOME
# sep_len_neural_net_model = CreateNeuralNetwork_SingleOutcome(
#     dataframe=iris,
#     outcome_variable='sepal length (cm)',
#     is_outcome_categorical=False,
#     list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#     number_of_hidden_layers=2,
#     scale_predictor_variables=True
# )
