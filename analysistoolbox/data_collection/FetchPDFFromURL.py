# Load packages
import requests

# Declare function
def FetchPDFFromURL(url, filename):
    """
    Download a PDF file from a URL and save it to the local filesystem.

    This function sends an HTTP GET request to the specified URL to retrieve a PDF document,
    then writes the binary content to a file with the specified filename. The function includes
    basic error handling to check if the download was successful (HTTP 200 status code) and
    provides feedback messages accordingly.

    Common use cases include:
      * Automating the collection of public reports or documents from government websites
      * Building document corpora for research or analysis
      * Batch downloading of PDFs for archival purposes
      * Creating local caches of remote PDF resources

    The function will save the file to the current working directory unless an absolute
    path is specified in the filename parameter.

    Parameters
    ----------
    url
        The URL of the PDF file to download. Must be a valid HTTP/HTTPS URL pointing
        to a PDF resource.
    filename
        The name (or path) of the file to save the downloaded PDF to. Should include
        the '.pdf' extension. Can be a relative filename (saves to current directory)
        or an absolute path.

    Returns
    -------
    None
        This function writes to a file and prints status messages, but does not
        return a value.

    Examples
    --------
    # Download a PDF from a URL and save it locally
    FetchPDFFromURL(
        url='https://example.com/document.pdf',
        filename='downloaded_document.pdf'
    )

    # Save to a specific directory using an absolute path
    FetchPDFFromURL(
        url='https://example.com/report.pdf',
        filename='/Users/username/Documents/report.pdf'
    )

    """
    # Lazy load uncommon packages
    import PyPDF2
    
    # Send a GET request to the url and get the response
    response = requests.get(url)
    
    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        # Open the filename in write-binary mode
        with open(filename, "wb") as file:
            # Write the response content to the file
            file.write(response.content)
            # Print a success message
            print(f"PDF downloaded and saved as {filename}")
    else:
        # Print an error message
        print(f"Failed to download PDF from {url}")

