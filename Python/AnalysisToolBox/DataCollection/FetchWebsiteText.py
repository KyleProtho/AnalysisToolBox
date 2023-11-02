# Load packages
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython.display import display, HTML, Markdown
import json
import os
import requests

# Declare function
def FetchWebsiteText(url,
                     browserless_api_key=None):
    """
    Fetches the text content from a specified website using the Browserless service.

    Parameters:
    url (str): The URL of the website from which to fetch text.
    browserless_api_key (str, optional): The API key for the Browserless service. If not provided, the function attempts to load it from a .env file.

    Returns:
    str: The text content of the website, with any occurrence of four or more newlines replaced with three newlines.
    """
    
    # If no API key is provided, try to load from .env file
    if browserless_api_key is None:
        load_dotenv()
        try:
            browserless_api_key = os.environ['BROWERLESS_API_KEY']
        except:
            raise ValueError("No API key provided and no .env file found. If you need a Browserless API key, visit https://www.browserless.io/")
    
    # Set payload to query
    payload = json.dumps({
        'url': url,
    })
    
    # Define headers for the API call
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }
    
    # Send the API call to browserless
    post_url = "https://chrome.browserless.io/content?token=" + browserless_api_key
    response = requests.request(
        method="POST", 
        url=post_url,
        headers=headers,
        data=payload
    )
    
    # Use BeautifulSoup to parse the response
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
    else:
        raise ValueError("Error fetching website contents. Status code: " + str(response.status_code))
    
    # Replace 4 or more newlines with 3 newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Return the text
    return text


# # Test the function
# contents = FetchWebsiteText(url="https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/#diverse-applications-implemented-with-autogen",
#                             browserless_api_key=open("C:/Users/oneno/OneDrive/Desktop/Browserless key.txt", "r").read())
