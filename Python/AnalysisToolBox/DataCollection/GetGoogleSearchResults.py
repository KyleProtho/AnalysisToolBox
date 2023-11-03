# Load packages
from dotenv import load_dotenv
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
    # If no API key is provided, load from .env file
    if serper_api_key is None:
        load_dotenv()
        try:
            serper_api_key = os.environ['SERPER_API_KEY']
        except:
            raise ValueError("No API key provided and no .env file found. If you need a Serper API key, visit https://serper.dev/")
    
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


# # Test the function
# results = GetGoogleSearchResults(
#     query='microsoft autogen',
#     serper_api_key=open("C:/Users/oneno/OneDrive/Desktop/Serper key.txt", 'r').read(),
#     display_results=True
# )
