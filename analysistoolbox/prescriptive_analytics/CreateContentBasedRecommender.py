# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

def CreateContentBasedRecommender(dataframe,
                                  outcome_variable,
                                  user_list_of_predictor_variables,
                                  item_list_of_predictor_variables,
                                  user_number_of_hidden_layers=2,
                                  item_number_of_hidden_layers=2,
                                  number_of_recommendations=10,
                                  test_size=0.2,
                                  scale_variables=True,
                                  plot_loss=True,
                                  plot_model_test_performance=True,
                                  print_peak_to_peak_range_of_each_predictor=False,
                                  initial_learning_rate=0.01,
                                  number_of_steps_gradient_descent=50,
                                  lambda_for_regularization=0.001,
                                  random_seed=412):
    """
    Create a content-based recommendation model using a two-tower neural network architecture.

    This function implements a "two-tower" or dual-encoder neural network to predict the
    strength of interaction between users and items based on their respective features.
    The architecture consists of two separate deep neural networks—one for user features
    and one for item features—that map both into a shared embedding space. The model
    learns to minimize the mean squared error between the dot product of these embeddings
    and the actual observed outcome (e.g., a rating or interaction score). This approach
    allows for flexible content-based filtering that generalizes across new users and
    items as long as their features are available.

    Content-based recommenders are essential for:
      * Matching patients to suitable clinical trials based on EHR and trial criteria
      * Recommending public health interventions to specific demographic segments
      * Suggesting relevant intelligence reports to analysts based on interest profiles
      * Personalized educational content delivery based on student learning styles
      * E-commerce product recommendations using item metadata and user browse history
      * Job-to-candidate matching using skill sets and job descriptions
      * Routing emergency resources based on geographic and situational metadata
      * Alerting intelligence officers to relevant temporal patterns in signal data

    The function handles feature scaling, train-test splitting, and provides visualizations
    for loss curves and model performance. It utilizes TensorFlow/Keras for the neural
    network implementation and L2 regularization to prevent overfitting.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset containing user features, item features, and the outcome variable.
    outcome_variable : str
        The name of the column representing the target interaction (e.g., 'rating', 'clicks').
    user_list_of_predictor_variables : list
        A list of column names representing user-specific attributes (e.g., 'age', 'location').
    item_list_of_predictor_variables : list
        A list of column names representing item-specific attributes (e.g., 'category', 'price').
    user_number_of_hidden_layers : int, optional
        Number of dense layers in the user feature tower. Defaults to 2.
    item_number_of_hidden_layers : int, optional
        Number of dense layers in the item feature tower. Defaults to 2.
    number_of_recommendations : int, optional
        The dimensionality of the shared embedding space (output of each tower). Defaults to 10.
    test_size : float, optional
        The proportion of data to reserve for the test set (0 to 1). Defaults to 0.2.
    scale_variables : bool, optional
        Whether to standardize input features and scale the outcome variable (MinMax to [-1, 1]).
        Highly recommended for neural network convergence. Defaults to True.
    plot_loss : bool, optional
        Whether to display a plot of the training loss over epochs. Defaults to True.
    plot_model_test_performance : bool, optional
        Whether to display a scatter/regression plot comparing predicted vs. actual outcomes.
        Defaults to True.
    print_peak_to_peak_range_of_each_predictor : bool, optional
        Whether to print the range (max - min) of each input variable. Defaults to False.
    initial_learning_rate : float, optional
        The learning rate for the Adam optimizer. Defaults to 0.01.
    number_of_steps_gradient_descent : int, optional
        The number of training epochs. Defaults to 50.
    lambda_for_regularization : float, optional
        L2 regularization penalty applied to dense layers. Defaults to 0.001.
    random_seed : int, optional
        Seed for reproducibility of weights and data splitting. Defaults to 412.

    Returns
    -------
    dict
        If `scale_variables` is True, returns a dictionary containing:
        - 'User Scaler': (StandardScaler) Fitted scaler for user features.
        - 'Item Scaler': (StandardScaler) Fitted scaler for item features.
        - 'Outcome Scaler': (MinMaxScaler) Fitted scaler for the outcome.
        - 'Model': (tf.keras.Model) The trained two-tower recommender model.
    tf.keras.Model
        If `scale_variables` is False, returns only the trained Keras model.

    Examples
    --------
    # Healthcare: Recommending clinical trials to patients
    import pandas as pd
    trial_data = pd.DataFrame({
        'patient_age': [25, 45, 65, 30, 55],
        'patient_severity': [2, 5, 8, 3, 6],
        'trial_phase': [1, 2, 3, 1, 2],
        'trial_risk_level': [0.1, 0.4, 0.7, 0.2, 0.5],
        'fit_score': [0.8, 0.6, 0.9, 0.7, 0.5]
    })
    results = CreateContentBasedRecommender(
        dataframe=trial_data,
        outcome_variable='fit_score',
        user_list_of_predictor_variables=['patient_age', 'patient_severity'],
        item_list_of_predictor_variables=['trial_phase', 'trial_risk_level'],
        number_of_steps_gradient_descent=100
    )

    # Intelligence: Suggesting surveillance targets to field officers
    surveillance_df = pd.DataFrame({
        'officer_experience': [5, 10, 2, 8, 15],
        'officer_specialty': [1, 3, 2, 1, 3],
        'target_volatility': [0.9, 0.2, 0.5, 0.8, 0.3],
        'target_distance': [10, 50, 5, 100, 20],
        'mission_success_prob': [0.9, 0.7, 0.4, 0.8, 0.9]
    })
    recommender_model = CreateContentBasedRecommender(
        dataframe=surveillance_df,
        outcome_variable='mission_success_prob',
        user_list_of_predictor_variables=['officer_experience', 'officer_specialty'],
        item_list_of_predictor_variables=['target_volatility', 'target_distance'],
        scale_variables=False  # Return the model directly
    )
    """
    # Lazy load uncommon packages
    import tensorflow as tf
    
    # Keep only the predictors from the user and item dataframes
    dataframe = dataframe[user_list_of_predictor_variables + item_list_of_predictor_variables + [outcome_variable]].copy()
    
    # Set the random seed
    tf.random.set_seed(random_seed)
    
    # Use Keras to create a normalization layer
    if scale_variables:
        # For users
        scalerUser = StandardScaler()
        scalerUser.fit(dataframe[user_list_of_predictor_variables])
        dataframe[user_list_of_predictor_variables] = scalerUser.transform(dataframe[user_list_of_predictor_variables])
        # For items
        scalerItem = StandardScaler()
        scalerItem.fit(dataframe[item_list_of_predictor_variables])
        dataframe[item_list_of_predictor_variables] = scalerItem.transform(dataframe[item_list_of_predictor_variables])
        # Outcome variable
        scalerOutcome = MinMaxScaler((-1, 1))
        scalerOutcome.fit(dataframe[outcome_variable].values.reshape(-1,1))
        dataframe[outcome_variable] = scalerOutcome.transform(dataframe[outcome_variable].values.reshape(-1,1))
        
    # Show the peak-to-peak range of each predictor
    if print_peak_to_peak_range_of_each_predictor:
        print("\nPeak-to-peak range of each predictor in user dataset:")
        print(np.ptp(dataframe[user_list_of_predictor_variables], axis=0))
        print("\nPeak-to-peak range of each predictor in item dataset:")
        print(np.ptp(dataframe[item_list_of_predictor_variables], axis=0))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        shuffle=True,
        random_state=random_seed
    )
    
    # Create dictionary of layers to be used in the neural network
    user_dict_layers = {}
    item_dict_layers = {}
    
    # Create hidden layers
    # # For users
    for i in range(user_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        if i < 10:
            layer_name = 'layer_0' + str(i + 1)
        else:
            layer_name = 'layer_' + str(i + 1)
        user_dict_layers[key_text] = tf.keras.layers.Dense(
            number_of_recommendations + ((user_number_of_hidden_layers - 1 - i) * 10),
            activation='relu', 
            name=layer_name,
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
    # # For items
    for i in range(item_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        if i < 10:
            layer_name = 'layer_0' + str(i + 1)
        else:
            layer_name = 'layer_' + str(i + 1)
        item_dict_layers[key_text] = tf.keras.layers.Dense(
            number_of_recommendations + ((item_number_of_hidden_layers - 1 - i) * 10),
            activation='relu', 
            name=layer_name,
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
        
    # Create output layer
    # # For users
    user_dict_layers['Output layer'] = tf.keras.layers.Dense(
        number_of_recommendations, 
        activation='linear', 
        name='final_layer',
        kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
    )
    # # For items
    item_dict_layers['Output layer'] = tf.keras.layers.Dense(
        number_of_recommendations, 
        activation='linear', 
        name='final_layer',
        kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
    )
    
    # Create list of layers to be used in the neural networks
    # # For users
    user_list_of_layers = []
    for i in range(user_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        user_list_of_layers.append(user_dict_layers[key_text])
    user_list_of_layers.append(user_dict_layers['Output layer'])
    # # For items
    item_list_of_layers = []
    for i in range(item_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        item_list_of_layers.append(item_dict_layers[key_text])
    item_list_of_layers.append(item_dict_layers['Output layer'])
    
    # Create sequential models
    user_model = tf.keras.Sequential(user_list_of_layers)
    item_model = tf.keras.Sequential(item_list_of_layers)
    
    # Create input layers and point to base layers
    # # For users
    user_input = tf.keras.layers.Input(shape=(len(user_list_of_predictor_variables),))
    user_output = user_model(user_input)
    user_output = tf.linalg.l2_normalize(user_output, axis=1)
    # # For items
    item_input = tf.keras.layers.Input(shape=(len(item_list_of_predictor_variables),))
    item_output = item_model(item_input)
    item_output = tf.linalg.l2_normalize(item_output, axis=1)
    
    # Create recommendation output layer
    recommender_output = tf.keras.layers.Dot(axes=1)([user_output, item_output])
    
    # Create model
    model = tf.keras.Model([user_input, item_input], recommender_output)
    
    # Show model summary
    model.summary()
    
    # Define loss function and optimizer
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        metrics=['mean_squared_error']
    )
    
    # Train the model
    loss_history = model.fit(
        [train[user_list_of_predictor_variables].values, train[item_list_of_predictor_variables].values],
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
        
    # # Evaluate the model
    # model.evaluate(
    #     [test[user_list_of_predictor_variables].values, test[item_list_of_predictor_variables].values],
    #     test[outcome_variable].values
    # )
    
    # Make predictions on test set
    predictions = model.predict([test[user_list_of_predictor_variables].values, test[item_list_of_predictor_variables].values])
    if scale_variables:
        predictions = scalerOutcome.inverse_transform(predictions.reshape(-1, 1))
        test[outcome_variable] = scalerOutcome.inverse_transform(test[outcome_variable].values.reshape(-1, 1))
    test['Predicted'] = predictions
    
    # Plot prediction performance if requested
    if plot_model_test_performance:
        plt.figure(figsize=(9,9))
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
            sns.regplot(
                data=test,
                x=outcome_variable,
                y='Predicted'
            )
            plt.plot(test[outcome_variable], test[outcome_variable], color='black', alpha=0.35)
            plt.title('Predicted vs. Observed Outcome', size = 15)
        plt.show()
        
    # Create dictionary of objects to be returned
    if scale_variables:
        dict_return = {
            'User Scaler': scalerUser,
            'Item Scaler': scalerItem,
            'Outcome Scaler': scalerOutcome,
            'Model': model
        }
        return(dict_return)
    else:
        return(model)

