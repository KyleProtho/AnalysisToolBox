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
import seaborn as sns
import sympy

# Set Seaborn style
sns.set(
    style="white",
    font="Arial",
    context="paper"
)

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
import seaborn as sns

# Set Seaborn style
sns.set(
    style="white",
    font="Arial",
    context="paper"
)

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
import seaborn as sns

# Set Seaborn style
sns.set(
    style="white",
    font="Arial",
    context="paper"
)

# Define observed and predicted values
observed_values = [1, 2, 3, 4, 5]
predicted_values = [1.1, 1.9, 3.2, 3.7, 5.1]

# Use the FindMinimumSquareLoss function
minimum_square_loss = FindMinimumSquareLoss(
    observed_values, 
    predicted_values, 
    show_plot=True
)

print(f"The minimum square loss is: {minimum_square_loss}")
```

#### PlotFunction

The **PlotFunction** function plots a mathematical function of x. It takes a lambda function as input and allows for customization of the plot.

```python
# Import the necessary libraries
from analysistoolbox.calculus import PlotFunction
import seaborn as sns

# Set Seaborn style
sns.set(
    style="white",
    font="Arial",
    context="paper"
)

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

# Define the path to the PDF file
filepath_to_pdf = "/path/to/your/input.pdf"

# Define the path to the text file
filepath_for_exported_text = "/path/to/your/output.txt"

# Call the function
ExtractTextFromPDF(
    filepath_to_pdf=filepath_to_pdf, 
    filepath_for_exported_text=filepath_for_exported_text, 
    start_page=1, 
    end_page=None
)
```

#### FetchPDFFromURL

The **FetchPDFFromURL** function downloads a PDF file from a URL and saves it to a specified location.

```python
from analysistoolbox.data_collection import FetchPDFFromURL

# URL of the PDF file to download
url = "https://example.com/sample.pdf"

# Name of the file to save the PDF as
filename = "sample.pdf"

# Call the function to download the PDF
FetchPDFFromURL(url, filename)
```

#### FetchUSShapefile

The **FetchUSShapefile** function fetches a geographical shapefile from the TIGER database of the U.S. Census Bureau. 

```python
from analysistoolbox.data_collection import FetchUSShapefile

# Fetch the shapefile for the census tracts in King County, Washington, for the 2021 census year
shapefile = FetchUSShapefile(state='WA', county='King', geography='tract', census_year=2021)

# Print the first few rows of the shapefile
print(shapefile.head())
```

#### FetchWebsiteText

The **FetchWebsiteText** function fetches the text from a website and saves it to a text file.

```python
# Import the function from the module
from analysistoolbox.data_collection import FetchWebsiteText

# Define the URL to fetch
url = "https://www.example.com"

# Define the Browserless API key
browserless_api_key = "your_browserless_api_key"

# Call the function
text = FetchWebsiteText(url, browserless_api_key)

# Print the fetched text
print(text)
```

#### GetGoogleSearchResults

The **GetGoogleSearchResults** function fetches Google search results for a given query using the Serper API.

```python
# Import the function
from analysistoolbox.data_collection import GetGoogleSearchResults

# Define a search query
query = "Python programming"

# Call the function with the query
# Make sure to replace 'your_serper_api_key' with your actual Serper API key
results = GetGoogleSearchResults(query, serper_api_key='your_serper_api_key', number_of_results=5, apply_autocorrect=True, display_results=True)

# Print the results
print(results)
```

#### GetZipFile

The **GetZipFile** function downloads a zip file from a url and saves it to a specified folder. It can also unzip the file and print the contents of the zip file.

```python
# Import the function
from analysistoolbox.data_collection import GetZipFile

# URL of the zip file to download
url = "http://example.com/file.zip"

# Path to the folder where the zip file will be saved
path_to_save_folder = "/path/to/save/folder"

# Call the function
GetZipFile(url, path_to_save_folder)
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
df = AddDateNumberColumns(df, 'Date')

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

# Print original dataframe
print("Original dataframe:")
print(df)

# Use the AddLeadingZeros function
df = AddLeadingZeros(df, 'ID', add_as_new_column=True)

# Print updated dataframe
print("Updated dataframe:")
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

# Define the parameters for the function
grouping_variables = ['Payment Method']
order_columns = ['Transaction Order']
ascending_order_args = [True]
row_count_column_name = 'Row Count'

# Call the function
new_df = AddRowCountColumn(df, grouping_variables, order_columns, ascending_order_args, row_count_column_name)
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
df_updated = AddTPeriodColumn(df, 'date', 'days')

# Print the updated dataframe
print(df_updated)
```

## Contributions

To report an issue, request a feature, or contribute to the project, please see the [CONTRIBUTING.md](CONTRIBUTING.md) file (in progress).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
