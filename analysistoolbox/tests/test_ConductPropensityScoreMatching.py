import pandas as pd
import numpy as np
from analysistoolbox.descriptive_analytics import ConductPropensityScoreMatching

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data
n_samples = 100
data = {
    'ID': range(1, n_samples + 1),
    'Age': np.random.normal(45, 15, n_samples),  # Age with mean 45, std 15
    'Income': np.random.normal(50000, 20000, n_samples),  # Income with mean 50k, std 20k
    'Education_Years': np.random.normal(16, 3, n_samples),  # Education years with mean 16, std 3
    'Treatment': np.random.choice(['Treatment', 'Control'], size=n_samples, p=[0.3, 0.7])  # 30% treatment, 70% control
}

# Create the dataframe
df = pd.DataFrame(data)

# Round numeric values for clarity
df['Age'] = df['Age'].round(1)
df['Income'] = df['Income'].round(0)
df['Education_Years'] = df['Education_Years'].round(1)

# Ensure we have enough control cases by adjusting some values
# This helps ensure we can find multiple matches
df.loc[df['Treatment'] == 'Control', 'Age'] = (
    df.loc[df['Treatment'] == 'Treatment', 'Age'].mean() + 
    np.random.normal(0, 2, len(df[df['Treatment'] == 'Control']))
)
df.loc[df['Treatment'] == 'Control', 'Income'] = (
    df.loc[df['Treatment'] == 'Treatment', 'Income'].mean() + 
    np.random.normal(0, 5000, len(df[df['Treatment'] == 'Control']))
)

# Conduct propensity score matching with 2 matches per subject
matched_df = ConductPropensityScoreMatching(
    dataframe=df,
    subject_id_column_name='ID',
    list_of_column_names_to_base_matching=['Age', 'Income', 'Education_Years'],
    grouping_column_name='Treatment',
    control_group_name='Control',
    max_matches_per_subject=2
)

# Display results
print("\nOriginal data shape:", df.shape)
print("Matched data shape:", matched_df.shape)
print("\nSample of matched pairs:")
sample_matches = matched_df[matched_df['Matched ID'].notna()].head(6)
print(sample_matches[['ID', 'Treatment', 'Age', 'Income', 'Education_Years', 'Matched ID', 'Propensity Score']])
