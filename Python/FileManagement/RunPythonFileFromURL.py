# Load packages
import requests

# Declare function
def RunPythonFileFromURL(python_url,
                         show_success_message=False):
    """This function will run a python file from a URL. This is useful for running python files from GitHub.

    Args:
        python_url (str): The raw Github url for the Python file. Must start with 'https://raw.githubusercontent.com'.
    """
    
    # Ensure that URL starts with 'https://raw.githubusercontent.com'
    if not python_url.startswith("https://raw.githubusercontent.com"):
        raise(Exception("URL must start with 'https://raw.githubusercontent.com'"))
    
    # Run python file from URL
    response = requests.get(python_url)
    
    # Extract the file name from the URL
    file_name = python_url.split('/')[-1]
    
    # Check if response is valid. If so, execute the python file
    if response.status_code == 200:
        with open('file.py', 'wb') as f:
            f.write(response.content)
        exec(open('file.py', encoding="utf-8").read())
        os.remove('file.py')
        
        # Make function within the file.py available outside of this function
        globals().update(locals())
        if show_success_message:
            print('Successfully ran ' + file_name + ' from URL!')


# # Test function
# RunPythonFileFromURL('https://raw.githubusercontent.com/KyleProtho/AnalysisToolBox/master/Python/Data%20Processing/CreateDataOverview.py')
# from sklearn import datasets
# import pandas as pd
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# CreateDataOverview(iris)

# RunPythonFileFromURL('https://raw.githubusercontent.com/KyleProtho/AnalysisToolBox/master/Python/Visualizations/PlotSingleVariableHistogram.py')
# PlotSingleVariableHistogram(
#     dataframe=iris,
#     quantitative_variable='sepal length (cm)'
# )
# CreateDataOverview(iris)
