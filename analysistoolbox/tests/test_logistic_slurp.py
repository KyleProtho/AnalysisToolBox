# Test script for CreateSLURPDistributionFromLogisticRegression
import pandas as pd
import numpy as np
import statsmodels.api as sm
from analysistoolbox.simulations.CreateSLURPDistributionFromLogisticRegression import CreateSLURPDistributionFromLogisticRegression

# Create a small test dataset
np.random.seed(42)
n_samples = 100

# Generate predictor variables
data = {
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'education_years': np.random.normal(14, 3, n_samples)
}

# Create a binary outcome based on the predictors
# Higher age, income, and education increase probability of success
log_odds = 0.1 * data['age'] + 0.00001 * data['income'] + 0.2 * data['education_years'] - 8
probabilities = 1 / (1 + np.exp(-log_odds))
data['success'] = np.random.binomial(1, probabilities, n_samples)

# Create DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("Test Dataset:")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"Success rate: {df['success'].mean():.3f}")

# Fit logistic regression model
X = df[['age', 'income', 'education_years']]
X = sm.add_constant(X)  # Add intercept
y = df['success']

# Fit the model
logistic_model = sm.Logit(y, X).fit()

# Display model summary
print("\nLogistic Regression Model Summary:")
print(logistic_model.summary())

# Test the SLURP function
print("\n" + "="*50)
print("Testing CreateSLURPDistributionFromLogisticRegression")
print("="*50)

# Test with specific predictor values
test_values = [35, 50000, 14]  # age=35, income=50000, education=14 years

# Debug: Check what get_prediction returns
print("Debugging prediction interval structure:")
pred = logistic_model.get_prediction([1] + test_values)
pred_interval = pred.summary_frame(alpha=0.05)
print("Available columns:", pred_interval.columns.tolist())
print("Prediction interval shape:", pred_interval.shape)
print("Prediction interval:\n", pred_interval)
print()

# Generate SLURP distribution
slurp_dist = CreateSLURPDistributionFromLogisticRegression(
    logistic_regression_model=logistic_model,
    list_of_prediction_values=test_values,
    number_of_trials=5000,  # Smaller number for quick testing
    prediction_interval=0.95,
    show_summary=True,
    title_for_plot="Probability Distribution Test",
    subtitle_for_plot=f"Age={test_values[0]}, Income=${test_values[1]:,}, Education={test_values[2]} years",
    caption_for_plot="Distribution of predicted probabilities based on logistic regression confidence intervals"
)

# Display results
print(f"\nSLURP Distribution Results:")

# Compare with direct prediction from the model
direct_pred = logistic_model.predict([1] + test_values)[0]
print(f"\nDirect model prediction: {direct_pred:.4f}")
print(f"SLURP mean vs direct prediction difference: {abs(slurp_dist['s_probability'].mean() - direct_pred):.4f}")
print(f"SLURP median vs direct prediction difference: {abs(slurp_dist['s_probability'].median() - direct_pred):.4f}")
print("\nTest completed successfully!")
