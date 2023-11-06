# Load packages
import requests
import PyPDF2

# Declare function
def FetchPDFFromURL(url, filename):
    """This function downloads a PDF from a website and saves it to the current directory.

    Args:
        url (str): The URL of the PDF file that you want to download.
        filename (str): The name of the file that you want to save the PDF to.
    """
    
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

