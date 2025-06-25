# Load packages
from analysistoolbox.simulations import CreateMetalogDistribution
import pandas as pd

# Create a test dataframe
test_df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5, 5, 5, 5, 5, 5],
})

# Test the CreateMetalogDistribution function
CreateMetalogDistribution(test_df, 'x')
CreateMetalogDistribution(test_df, 'x', lower_bound=0, upper_bound=6)

