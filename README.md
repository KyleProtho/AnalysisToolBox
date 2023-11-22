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
import pandas as pd
from analysistoolbox.data_processing import CleanTextColumns

# Create a sample dataframe
df = pd.DataFrame({
    'A': [' hello', 'world ', ' python '],
    'B': [1, 2, 3],
})

# Clean the dataframe
df_clean = CleanTextColumns(df)

# Print the updated dataframe
print(df_clean)
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

## Contributions

To report an issue, request a feature, or contribute to the project, please see the [CONTRIBUTING.md](CONTRIBUTING.md) file (in progress).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
