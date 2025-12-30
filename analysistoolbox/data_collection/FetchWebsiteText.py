# Load packages
from bs4 import BeautifulSoup
from IPython.display import display, HTML, Markdown
import json
import os
import requests

# Declare function
def FetchWebsiteText(url,
                     browserless_api_key=None):
    """
    Fetch and extract text content from a website using the Browserless service.

    This function retrieves the HTML content of a specified URL using the Browserless service,
    a headless browser automation platform. Unlike simple HTTP requests, Browserless can
    execute JavaScript and render dynamic content, making it ideal for modern websites that
    rely on client-side rendering frameworks (React, Vue, Angular, etc.).

    The function then parses the HTML using BeautifulSoup to extract clean text content,
    automatically removing HTML tags and formatting. Excessive whitespace (4+ consecutive
    newlines) is normalized to improve readability. The Browserless API key can be provided
    directly or loaded from environment variables for security.

    Common use cases include:
      * Web scraping for research and data collection
      * Content monitoring and change detection
      * Competitive intelligence gathering
      * Automated content extraction for text analysis or NLP
      * Archiving web content for documentation
      * Extracting text from JavaScript-heavy websites

    Note: Using this function requires a valid Browserless API key, which can be obtained
    from browserless.io. Respect website terms of service and robots.txt when scraping.

    Parameters
    ----------
    url
        The URL of the website from which to fetch text content. Must be a valid HTTP/HTTPS URL.
    browserless_api_key
        API key for the Browserless service. If not provided, the function attempts to load
        it from environment variables (typically stored in a .env file). Defaults to None.

    Returns
    -------
    str
        The extracted text content of the website, with HTML tags removed and excessive
        newlines (4 or more) normalized to three newlines for improved readability.

    Examples
    --------
    # Fetch text from a website with API key provided directly
    text = FetchWebsiteText(
        url='https://example.com/article',
        browserless_api_key='your_api_key_here'
    )

    # Fetch text using API key from environment variables
    # (assuming BROWSERLESS_API_KEY is set in .env file)
    import os
    api_key = os.getenv('BROWSERLESS_API_KEY')
    text = FetchWebsiteText(
        url='https://example.com/dynamic-page',
        browserless_api_key=api_key
    )

    """
    
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

