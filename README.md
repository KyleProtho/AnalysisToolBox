# Analysis Tool Box

## Description

Analysis Tool Box (i.e. "analysistoolbox") is a collection of tools in Python for data collection and processing, statisitics, analytics, and intelligence analysis.

## Getting Started

To install the package, run the following command in the root directory of the project:

```bash
pip install analysistoolbox
```

Visualizations are created using the matplotlib and seaborn libraries. While you can select whichever seaborn style you'd like, the following Seaborn style tends to get the best looking plots:

```python
sns.set(
    style="white",
    font="Arial",
    context="paper"
)
```

## Usage

There are many modules in the analysistoolbox package, each with their own functions. The following is a list of the modules:

* Calculus
* Data collection
* Data processing
* Descriptive analytics
* File management
* Hypothesis testing
* Linear algebra
* Predictive analytics
* Statistics
* Visualizations

### Calculus

There are several functions in the Calculus submodule. The following is a list of the functions:

* FindDerivative
* FindLimitOfFunction
* FindMinimumSquareLoss
* PlotFunction

#### FindDerivative

The **FindDerivative** function calculates the derivative of a given function. It uses the sympy library, a Python library for symbolic mathematics, to perform the differentiation. The function also has the capability to print the original function and its derivative, return the derivative function, and plot both the original function and its derivative.

```python
# Load the FindDerivative function from the Calculus submodule
from analysistoolbox.calculus import FindDerivative
import sympy

# Define a symbolic variable
x = sympy.symbols('x')

# Define a function
f_of_x = x**3 + 2*x**2 + 3*x + 4

# Use the FindDerivative function
FindDerivative(
    f_of_x, 
    print_functions=True, 
    return_derivative_function=True, 
    plot_functions=True
)
```

#### FindLimitOfFunction

The **FindLimitOfFunction** function finds the limit of a function at a specific point and optionally plot the function and its tangent line at that point. The script uses the matplotlib and numpy libraries for plotting and numerical operations respectively.

```python
# Import the necessary libraries
from analysistoolbox.calculus import FindLimitOfFunction
import numpy as np
import sympy

# Define a symbolic variable
x = sympy.symbols('x')

# Define a function
f_of_x = np.sin(x) / x

# Use the FindLimitOfFunction function
FindLimitOfFunction(
    f_of_x, 
    point=0, 
    step=0.01, 
    plot_function=True, 
    x_minimum=-10, 
    x_maximum=10, 
    n=1000, 
    tangent_line_window=1
)
```

#### FindMinimumSquareLoss

The **FindMinimumSquareLoss** function calculates the minimum square loss between observed and predicted values. This function is often used in machine learning and statistics to measure the average squared difference between the actual and predicted outcomes.

```python
# Import the necessary libraries
from analysistoolbox.calculus import FindMinimumSquareLoss

# Define observed and predicted values
observed_values = [1, 2, 3, 4, 5]
predicted_values = [1.1, 1.9, 3.2, 3.7, 5.1]

# Use the FindMinimumSquareLoss function
minimum_square_loss = FindMinimumSquareLoss(
    observed_values, 
    predicted_values, 
    show_plot=True
)

# Print the minimum square loss
print(f"The minimum square loss is: {minimum_square_loss}")
```

#### PlotFunction

The **PlotFunction** function plots a mathematical function of x. It takes a lambda function as input and allows for customization of the plot.

```python
# Import the necessary libraries
from analysistoolbox.calculus import PlotFunction
import sympy

# Set x as a symbolic variable
x = sympy.symbols('x')

# Define the function to plot
f_of_x = lambda x: x**2

# Plot the function with default settings
PlotFunction(f_of_x)
```

### Data Collection

There are several functions in the Data Collection submodule. The following is a list of the functions:

* ExtractTextFromPDF
* FetchPDFFromURL
* FetchUSShapefile
* FetchWebsiteText
* GetCompanyFilings
* GetGoogleSearchResults
* GetZipFile

#### ExtractTextFromPDF

The **ExtractTextFromPDF** function extracts text from a PDF file, cleans it, then saves it to a text file.

```python
# Import the function
from analysistoolbox.data_collection import ExtractTextFromPDF

# Call the function
ExtractTextFromPDF(
    filepath_to_pdf="/path/to/your/input.pdf", 
    filepath_for_exported_text="/path/to/your/output.txt", 
    start_page=1, 
    end_page=None
)
```

#### FetchPDFFromURL

The **FetchPDFFromURL** function downloads a PDF file from a URL and saves it to a specified location.

```python
# Import the function
from analysistoolbox.data_collection import FetchPDFFromURL

# Call the function to download the PDF
FetchPDFFromURL(
    url="https://example.com/sample.pdf", 
    filename="C:/folder/sample.pdf"
)
```

#### FetchUSShapefile

The **FetchUSShapefile** function fetches a geographical shapefile from the TIGER database of the U.S. Census Bureau. 

```python
# Import the function
from analysistoolbox.data_collection import FetchUSShapefile

# Fetch the shapefile for the census tracts in King County, Washington, for the 2021 census year
shapefile = FetchUSShapefile(
    state='PA', 
    county='Allegheny', 
    geography='tract', 
    census_year=2021
)

# Print the first few rows of the shapefile
print(shapefile.head())
```

#### FetchWebsiteText

The **FetchWebsiteText** function fetches the text from a website and saves it to a text file.

```python
# Import the function
from analysistoolbox.data_collection import FetchWebsiteText

# Call the function
text = FetchWebsiteText(
    url="https://www.example.com", 
    browserless_api_key="your_browserless_api_key"
)

# Print the fetched text
print(text)
```

#### GetCompanyFilings

The **GetCompanyFilings** function fetches company filings from the SEC EDGAR database. It returns a list of filings for a given company CIK (Central Index Key) and filing type.

```python
# Import the function
from analysistoolbox.data_collection import GetCompanyFilings

# Call the function to get company filings for 'Online Dating' companies in 2024
results = GetCompanyFilings(
        search_keywords="Online Dating",
        start_date="2024-01-01",
        end_date="2024-12-31",
        filing_type="all",
    )

# Print the results
print(results)
```

#### GetGoogleSearchResults

The **GetGoogleSearchResults** function fetches Google search results for a given query using the Serper API.

```python
# Import the function
from analysistoolbox.data_collection import GetGoogleSearchResults

# Call the function with the query
# Make sure to replace 'your_serper_api_key' with your actual Serper API key
results = GetGoogleSearchResults(
    query="Python programming", 
    serper_api_key='your_serper_api_key', 
    number_of_results=5, 
    apply_autocorrect=True, 
    display_results=True
)

# Print the results
print(results)
```

#### GetZipFile

The **GetZipFile** function downloads a zip file from a url and saves it to a specified folder. It can also unzip the file and print the contents of the zip file.

```python
# Import the function
from analysistoolbox.data_collection import GetZipFile

# Call the function
GetZipFile(
    url="http://example.com/file.zip", 
    path_to_save_folder="/path/to/save/folder"
)
```

### Data Processing

There are several functions in the Data Processing submodule. The following is a list of the functions:

* AddDateNumberColumns
* AddLeadingZeros
* AddRowCountColumn
* AddTPeriodColumn
* AddTukeyOutlierColumn
* CleanTextColumns
* ConductAnomalyDetection
* ConductEntityMatching
* ConvertOddsToProbability
* CountMissingDataByGroup
* CreateBinnedColumn
* CreateDataOverview
* CreateRandomSampleGroups
* CreateRareCategoryColumn
* CreateStratifiedRandomSampleGroups
* ImputeMissingValuesUsingNearestNeighbors
* VerifyGranularity

#### AddDateNumberColumns

The **AddDateNumberColumns** function adds columns for the year, month, quarter, week, day of the month, and day of the week to a dataframe.

```python
# Import necessary packages
from analysistoolbox.data_processing import AddDateNumberColumns
from datetime import datetime
import pandas as pd

# Create a sample dataframe
data = {'Date': [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1), datetime(2020, 4, 1)]}
df = pd.DataFrame(data)

# Use the function on the sample dataframe
df = AddDateNumberColumns(
    dataframe=df, 
    date_column_name='Date'
)

# Print the updated dataframe
print(df)
```

#### AddLeadingZeros

The **AddLeadingZeros** function adds leading zeros to a column. If fixed_length is not specified, the longest string in the column is used as the fixed length. If add_as_new_column is set to True, the new column is added to the dataframe. Otherwise, the original column is updated.

```python
# Import necessary packages
from analysistoolbox.data_processing import AddLeadingZeros
import pandas as pd

# Create a sample dataframe
data = {'ID': [1, 23, 456, 7890]}
df = pd.DataFrame(data)

# Use the AddLeadingZeros function
df = AddLeadingZeros(
    dataframe=df, 
    column_name='ID', 
    add_as_new_column=True
)

# Print updated dataframe
print(df)
```

#### AddRowCountColumn

The **AddRowCountColumn** function adds a column to a dataframe that contains the row number for each row, based on a group (or groups) of columns. The function can also sort the dataframe by a column or columns before adding the row count column.

```python
# Import necessary packages
from analysistoolbox.data_processing import AddRowCountColumn
import pandas as pd

# Create a sample dataframe
data = {
    'Payment Method': ['Check', 'Credit Card', 'Check', 'Credit Card', 'Check', 'Credit Card', 'Check', 'Credit Card'],
    'Transaction Value': [100, 200, 300, 400, 500, 600, 700, 800],
    'Transaction Order': [1, 2, 3, 4, 5, 6, 7, 8]
}
df = pd.DataFrame(data)

# Call the function
df_updated = AddRowCountColumn(
    dataframe=df, 
    list_of_grouping_variables=['Payment Method'], 
    list_of_order_columns=['Transaction Order'], 
    list_of_ascending_order_args=[True]
)

# Print the updated dataframe
print(df_updated)
```

#### AddTPeriodColumn

The **AddTPeriodColumn** function adds a T-period column to a dataframe. The T-period column is the number of intervals (e.g., days or weeks) since the earliest date in the dataframe.

```python
# Import necessary libraries
from analysistoolbox.data_processing import AddTPeriodColumn
from datetime import datetime
import pandas as pd

# Create a sample dataframe
data = {
    'date': pd.date_range(start='1/1/2020', end='1/10/2020'),
    'value': range(1, 11)
}
df = pd.DataFrame(data)

# Use the function
df_updated = AddTPeriodColumn(
    dataframe=df, 
    date_column_name='date', 
    t_period_interval='days'
)

# Print the updated dataframe
print(df_updated)
```

#### AddTukeyOutlierColumn

The **AddTukeyOutlierColumn** function adds a column to a dataframe that indicates whether a value is an outlier. The function uses the Tukey method to identify outliers.

```python
# Import necessary libraries
from analysistoolbox.data_processing import AddTukeyOutlierColumn
import pandas as pd

# Create a sample dataframe
data = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]})

# Use the function
df_updated = AddTukeyOutlierColumn(
    dataframe=data, 
    value_column_name='values', 
    tukey_boundary_multiplier=1.5, 
    plot_tukey_outliers=True
)

# Print the updated dataframe
print(df_updated)
```

#### CleanTextColumns

The **CleanTextColumns** function cleans string-type columns in a pandas DataFrame by removing all leading and trailing spaces.

```python
# Import necessary libraries
from analysistoolbox.data_processing import CleanTextColumns
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({
    'A': [' hello', 'world ', ' python '],
    'B': [1, 2, 3],
})

# Clean the dataframe
df_clean = CleanTextColumns(df)
```

#### ConductAnomalyDetection

The **ConductAnomalyDetection** function performs anomaly detection on a given dataset using the z-score method.

```python
# Import necessary libraries
from analysistoolbox.data_processing import ConductAnomalyDetection
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({
    'A': [1, 2, 3, 1000],
    'B': [4, 5, 6, 2000],
})

# Conduct anomaly detection
df_anomaly_detected = ConductAnomalyDetection(
    dataframe=df, 
    list_of_columns_to_analyze=['A', 'B']
)

# Print the updated dataframe
print(df_anomaly_detected)
```

#### ConductEntityMatching

The **ConductEntityMatching** function performs entity matching between two dataframes using various fuzzy matching algorithms.

```python
from analysistoolbox.data_processing import ConductEntityMatching
import pandas as pd

# Create two dataframes
dataframe_1 = pd.DataFrame({
    'ID': ['1', '2', '3'],
    'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'City': ['New York', 'Los Angeles', 'Chicago']
})

dataframe_2 = pd.DataFrame({
    'ID': ['a', 'b', 'c'],
    'Name': ['Jon Doe', 'Jane Smyth', 'Robert Johnson'],
    'City': ['NYC', 'LA', 'Chicago']
})

# Conduct entity matching
matched_entities = ConductEntityMatching(
    dataframe_1=dataframe_1,
    dataframe_1_primary_key='ID',
    dataframe_2=dataframe_2,
    dataframe_2_primary_key='ID',
    levenshtein_distance_filter=3,
    match_score_threshold=80,
    columns_to_compare=['Name', 'City'],
    match_methods=['Partial Token Set Ratio', 'Weighted Ratio']
)
``` 

#### ConvertOddsToProbability

The **ConvertOddsToProbability** function converts odds to probability in a new column.

```python
# Import necessary packages
from analysistoolbox.data_processing import ConvertOddsToProbability
import pandas as pd

# Create a sample dataframe
data = {
    'Team': ['Team1', 'Team2', 'Team3', 'Team4'],
    'Odds': [2.5, 1.5, 3.0, np.nan]
}
df = pd.DataFrame(data)

# Print the original dataframe
print("Original DataFrame:")
print(df)

# Use the function to convert odds to probability
df = ConvertOddsToProbability(
    dataframe=df, 
    odds_column='Odds'
)
```

#### CountMissingDataByGroup

The **CountMissingDataByGroup** function counts the number of records with missing data in a Pandas dataframe, grouped by specified columns.

```python
# Import necessary packages
from analysistoolbox.data_processing import CountMissingDataByGroup
import pandas as pd
import numpy as np

# Create a sample dataframe with some missing values
data = {
    'Group': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value1': [1, 2, np.nan, 4, 5, np.nan],
    'Value2': [np.nan, 8, 9, 10, np.nan, 12]
}
df = pd.DataFrame(data)

# Use the function to count missing data by group
CountMissingDataByGroup(
    dataframe=df, 
    list_of_grouping_columns=['Group']
)
```

#### CreateBinnedColumn

The **CreateBinnedColumn** function creates a new column in a Pandas dataframe based on a numeric variable. Binning is a process of transforming continuous numerical variables into discrete categorical 'bins'.

```python
# Import necessary packages
from analysistoolbox.data_processing import CreateBinnedColumn
import pandas as pd
import numpy as np

# Create a sample dataframe
data = {
    'Group': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value1': [1, 2, 3, 4, 5, 6],
    'Value2': [7, 8, 9, 10, 11, 12]
}
df = pd.DataFrame(data)

# Use the function to create a binned column
df_binned = CreateBinnedColumn(
    dataframe=df, 
    numeric_column_name='Value1', 
    number_of_bins=3, 
    binning_strategy='uniform'
)
```

#### CreateDataOverview

The **CreateDataOverview** function creates an overview of a Pandas dataframe, including the data type, missing count, missing percentage, and summary statistics for each variable in the DataFrame.

```python
# Import necessary packages
from analysistoolbox.data_processing import CreateDataOverview
import pandas as pd
import numpy as np

# Create a sample dataframe
data = {
    'Column1': [1, 2, 3, np.nan, 5, 6],
    'Column2': ['a', 'b', 'c', 'd', np.nan, 'f'],
    'Column3': [7.1, 8.2, 9.3, 10.4, np.nan, 12.5]
}
df = pd.DataFrame(data)

# Use the function to create an overview of the dataframe
CreateDataOverview(
    dataframe=df, 
    plot_missingness=True
)
```

#### CreateRandomSampleGroups

The **CreateRandomSampleGroups** function a takes a pandas DataFrame, shuffle its rows, assign each row to one of n groups, and then return the updated DataFrame with an additional column indicating the group number.

```python
# Import necessary packages
from analysistoolbox.data_processing import CreateRandomSampleGroups 
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 31, 35, 19, 45],
    'Score': [85, 95, 78, 81, 92]
}
df = pd.DataFrame(data)

# Use the function
grouped_df = CreateRandomSampleGroups(
    dataframe=df, 
    number_of_groups=2, 
    random_seed=123
)
```

#### CreateRareCategoryColumn

The **CreateRareCategoryColumn** function creates a new column in a Pandas dataframe that indicates whether a categorical variable value is rare. A rare category is a category that occurs less than a specified percentage of the time.

```python
# Import necessary packages
from analysistoolbox.data_processing import CreateRareCategoryColumn 
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice', 'Bob', 'Alice'],
    'Age': [25, 31, 35, 19, 45, 23, 30, 24],
    'Score': [85, 95, 78, 81, 92, 88, 90, 86]
}
df = pd.DataFrame(data)

# Use the function
updated_df = CreateRareCategoryColumn(
    dataframe=df, 
    categorical_column_name='Name', 
    rare_category_label='Rare', 
    rare_category_threshold=0.05,
    new_column_suffix='(rare category)'
)
```

#### CreateStratifiedRandomSampleGroups

The **CreateStratifiedRandomSampleGroups** unction performs stratified random sampling on a pandas DataFrame. Stratified random sampling is a method of sampling that involves the division of a population into smaller groups known as strata. In stratified random sampling, the strata are formed based on members' shared attributes or characteristics.

```python
# Import necessary packages
from analysistoolbox.data_processing import CreateStratifiedRandomSampleGroups
import numpy as np
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice', 'Bob', 'Alice'],
    'Age': [25, 31, 35, 19, 45, 23, 30, 24],
    'Score': [85, 95, 78, 81, 92, 88, 90, 86]
}
df = pd.DataFrame(data)

# Use the function
stratified_df = CreateStratifiedRandomSampleGroups(
    dataframe=df, 
    number_of_groups=2, 
    list_categorical_column_names=['Name'], 
    random_seed=42
)
```

#### ImputeMissingValuesUsingNearestNeighbors

The **ImputeMissingValuesUsingNearestNeighbors** function imputes missing values in a dataframe using the nearest neighbors method. For each sample with missing values, it finds the n_neighbors nearest neighbors in the training set and imputes the missing values using the mean value of these neighbors.

```python
# Import necessary packages
from analysistoolbox.data_processing import ImputeMissingValuesUsingNearestNeighbors
import pandas as pd
import numpy as np

# Create a sample DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5],
    'D': [1, 2, 3, 4, np.nan]
}
df = pd.DataFrame(data)

# Use the function
imputed_df = ImputeMissingValuesUsingNearestNeighbors(
    dataframe=df, 
    list_of_numeric_columns_to_impute=['A', 'B', 'C', 'D'], 
    number_of_neighbors=2, 
    averaging_method='uniform'
)
```

#### VerifyGranularity

The **VerifyGranularity** function checks the granularity of a given dataframe based on a list of key columns. Granularity in this context refers to the level of detail or summarization in a set of data.

```python
# Import necessary packages
from analysistoolbox.data_processing import VerifyGranularity
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice', 'Bob', 'Alice'],
    'Age': [25, 31, 35, 19, 45, 23, 30, 24],
    'Score': [85, 95, 78, 81, 92, 88, 90, 86]
}
df = pd.DataFrame(data)

# Use the function
VerifyGranularity(
    dataframe=df, 
    list_of_key_columns=['Name', 'Age'], 
    set_key_as_index=True, 
    print_as_markdown=False
)
```

### Descriptive Analytics

There are several functions in the Descriptive Analytics submodule. The following is a list of the functions:

* ConductManifoldLearning
* ConductPrincipalComponentAnalysis
* CreateAssociationRules
* CreateGaussianMixtureClusters
* CreateHierarchicalClusters
* CreateKMeansClusters
* GenerateEDAWithLIDA

#### ConductManifoldLearning

The **ConductManifoldLearning** function performs manifold learning on a given dataframe and returns a new dataframe with the original columns and the new manifold learning components. Manifold learning is a type of unsupervised learning that is used to reduce the dimensionality of the data.

```python
# Import necessary packages
from analysistoolbox.descriptive_analytics import ConductManifoldLearning
import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Use the function
new_df = ConductManifoldLearning(
    dataframe=iris_df, 
    list_of_numeric_columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], 
    number_of_components=2, 
    random_seed=42, 
    show_component_summary_plots=True, 
    summary_plot_size=(10, 10)
)
```

#### ConductPrincipalComponentAnalysis

The **ConductPrincipalComponentAnalysis** function performs Principal Component Analysis (PCA) on a given dataframe. PCA is a technique used in machine learning to reduce the dimensionality of data while retaining as much information as possible.

```python
# Import necessary packages
from analysistoolbox.descriptive_analytics import ConductManifoldLearning
import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Call the function
result = ConductPrincipalComponentAnalysis(
    dataframe=iris_df,
    list_of_numeric_columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    number_of_components=2
)
```

#### CreateAssociationRules

The **CreateAssociationRules** function creates association rules from a given dataframe. Association rules are widely used in market basket analysis, where the goal is to find associations and/or correlations among a set of items.

```python
# Import necessary packages
from analysistoolbox.descriptive_analytics import CreateAssociationRules
import pandas as pd

# Assuming you have a dataframe 'df' with 'TransactionID' and 'Item' columns
result = CreateAssociationRules(
    dataframe=df,
    transaction_id_column='TransactionID',
    items_column='Item',
    support_threshold=0.01,
    confidence_threshold=0.2,
    plot_lift=True,
    plot_title='Association Rules',
    plot_size=(10, 7)
)
```

#### CreateGaussianMixtureClusters

The **CreateGaussianMixtureClusters** function creates Gaussian mixture clusters from a given dataframe. Gaussian mixture models are a type of unsupervised learning that is used to find clusters in data. It adds the resulting clusters as a new column in the dataframe, and also calculates the probability of each data point belonging to each cluster.

```python
# Import necessary packages
from analysistoolbox.descriptive_analytics import CreateGaussianMixtureClusters
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()

# Convert the iris dataset to a pandas dataframe
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

# Call the CreateGaussianMixtureClusters function
df_clustered = CreateGaussianMixtureClusters(
    dataframe=df,
    list_of_numeric_columns_for_clustering=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    number_of_clusters=3,
    column_name_for_clusters='Gaussian Mixture Cluster',
    scale_predictor_variables=True,
    show_cluster_summary_plots=True,
    sns_color_palette='Set2',
    summary_plot_size=(15, 15),
    random_seed=123,
    maximum_iterations=200
)
```

#### CreateHierarchicalClusters

The **CreateHierarchicalClusters** function creates hierarchical clusters from a given dataframe. Hierarchical clustering is a type of unsupervised learning that is used to find clusters in data. It adds the resulting clusters as a new column in the dataframe.

```python
# Import necessary packages
from analysistoolbox.descriptive_analytics import CreateHierarchicalClusters
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Call the CreateHierarchicalClusters function
df_clustered = CreateHierarchicalClusters(
    dataframe=df,
    list_of_value_columns_for_clustering=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    number_of_clusters=3,
    column_name_for_clusters='Hierarchical Cluster',
    scale_predictor_variables=True,
    show_cluster_summary_plots=True,
    color_palette='Set2',
    summary_plot_size=(6, 4),
    random_seed=412,
    maximum_iterations=300
)
```

#### CreateKMeansClusters

The **CreateKMeansClusters** function performs K-Means clustering on a given dataset and returns the dataset with an additional column indicating the cluster each record belongs to.

```python
# Import necessary packages
from analysistoolbox.descriptive_analytics import CreateKMeansClusters
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Call the CreateKMeansClusters function
df_clustered = CreateKMeansClusters(
    dataframe=df,
    list_of_value_columns_for_clustering=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    number_of_clusters=3,
    column_name_for_clusters='K-Means Cluster',
    scale_predictor_variables=True,
    show_cluster_summary_plots=True,
    color_palette='Set2',
    summary_plot_size=(6, 4),
    random_seed=412,
    maximum_iterations=300
)
```

#### GenerateEDAWithLIDA

The **GenerateEDAWithLIDA** function uses the LIDA package from Microsoft to generate exploratory data analysis (EDA) goals. 

```python
# Import necessary packages
from analysistoolbox.descriptive_analytics import GenerateEDAWithLIDA
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Call the GenerateEDAWithLIDA function
df_summary = GenerateEDAWithLIDA(
    dataframe=df,
    llm_api_key="your_llm_api_key_here",
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    visualization_library="seaborn",
    goal_temperature=0.50,
    code_generation_temperature=0.05,
    data_summary_method="llm",
    number_of_samples_to_show_in_summary=5,
    return_data_fields_summary=True,
    number_of_goals_to_generate=5,
    plot_recommended_visualization=True,
    show_code_for_recommended_visualization=True
)
```

### File Management

### Hypothesis Testing

### Linear Algebra

### Predictive Analytics

### Prescriptive Analytics

### Simulations

### Visualizations

## Contributions

To report an issue, request a feature, or contribute to the project, please see the [CONTRIBUTING.md](CONTRIBUTING.md) file (in progress).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
