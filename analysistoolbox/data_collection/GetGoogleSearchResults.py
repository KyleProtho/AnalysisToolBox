# Load packages
from IPython.display import display, HTML, Markdown
import json
import os
import requests

# Declare function
def GetGoogleSearchResults(query,
                           serper_api_key=None,
                           number_of_results=10,
                           apply_autocorrect=False,
                           display_results=False):
    """
    Fetches Google search results for a given query using the Serper API.

    Args:
        query (str): The search query to get results for.
        serper_api_key (str, optional): The API key for the Serper service. If not provided, it will try to load from .env file. Defaults to None.
        number_of_results (int, optional): The number of search results to return. Defaults to 10.
        apply_autocorrect (bool, optional): Whether to apply autocorrect to the query. Defaults to False.
        display_results (bool, optional): Whether to display the results in a DataFrame. Defaults to False.

    Raises:
        ValueError: If no API key is provided and no .env file is found.

    Returns:
        dict: The response from the Serper API, containing the search results.
    """
    
    # Set Google Search URL
    url = "https://google.serper.dev/search"
    
    # Set payload to query
    payload = json.dumps({
        'q': query,
        "autocorrect": apply_autocorrect,
        "num": number_of_results
    })
    
    # Define headers for API call
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    
    # Get the response from the API
    response = requests.request(
        method="POST", 
        url=url, 
        headers=headers, 
        data=payload
    )
    
    # Convert the response to json
    response = response.json()
    
    # Display the results if requested
    if display_results:
        # Create a dataframe of the organic results
        results_df = pd.DataFrame(results['organic'])
        # Display the dataframe
        display(results_df)
        
    # Return the response
    return response

