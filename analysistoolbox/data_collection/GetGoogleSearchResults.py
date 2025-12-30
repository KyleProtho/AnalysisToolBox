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
    Fetch Google search results programmatically using the Serper API.

    This function retrieves Google search results for a specified query using the Serper API,
    a service that provides programmatic access to Google Search without the need for web
    scraping or managing headless browsers. The API returns structured JSON data containing
    organic search results, including titles, snippets, URLs, and ranking positions.

    Unlike direct web scraping, using the Serper API provides:
      * Reliable, structured data without HTML parsing
      * No risk of IP blocking or CAPTCHA challenges
      * Consistent data format across queries
      * Access to Google's autocorrect and spell-check features
      * Support for large-scale search operations

    Common use cases include:
      * Search engine optimization (SEO) research and competitive analysis
      * Brand monitoring and reputation management
      * Market research and trend analysis
      * Content discovery and curation
      * Academic research on search behavior and information retrieval
      * Automated fact-checking and source verification
      * Lead generation and business intelligence

    The function optionally displays results as a DataFrame for quick inspection and
    supports autocorrect to handle misspelled queries.

    Parameters
    ----------
    query
        The search query string to submit to Google. Supports all standard Google search
        operators (e.g., quotes for exact match, site: for domain filtering).
    serper_api_key
        API key for the Serper service. If not provided, the function attempts to load
        it from environment variables (typically in a .env file). Defaults to None.
    number_of_results
        Number of search results to return. Maximum depends on API subscription tier.
        Defaults to 10.
    apply_autocorrect
        If True, applies Google's autocorrect feature to fix misspelled queries. If False,
        searches for the exact query as provided. Defaults to False.
    display_results
        If True, displays the organic search results in a pandas DataFrame for immediate
        inspection in notebook environments. Defaults to False.

    Returns
    -------
    dict
        The JSON response from the Serper API containing search results. Key fields include:
        'organic' (list of search results), 'searchParameters' (query metadata), and
        optionally 'relatedSearches', 'knowledgeGraph', etc.

    Raises
    ------
    ValueError
        If no API key is provided and it cannot be loaded from environment variables.

    Examples
    --------
    # Basic search with 10 results
    results = GetGoogleSearchResults(
        query='machine learning tutorials',
        serper_api_key='your_api_key_here'
    )

    # Search with autocorrect enabled and display results
    results = GetGoogleSearchResults(
        query='artifical inteligence',  # misspelled
        serper_api_key='your_api_key_here',
        apply_autocorrect=True,
        display_results=True
    )

    # Get more results for comprehensive analysis
    results = GetGoogleSearchResults(
        query='climate change research site:edu',
        serper_api_key='your_api_key_here',
        number_of_results=50
    )

    # Use API key from environment variables
    import os
    api_key = os.getenv('SERPER_API_KEY')
    results = GetGoogleSearchResults(
        query='python data science',
        serper_api_key=api_key,
        number_of_results=20
    )

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

