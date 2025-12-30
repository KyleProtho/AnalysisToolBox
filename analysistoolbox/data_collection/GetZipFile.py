# Load packages
import io
import requests

# Declare function
def GetZipFile(url,
               path_to_save_folder,
               unzip=True,
               print_contents=True):
    """
    Download and extract a ZIP file from a URL to a local directory.

    This function retrieves a ZIP archive from a specified URL via HTTP GET request,
    optionally extracts all contents to a target directory, and can display the list
    of files contained within the archive. The function handles the download in memory
    without creating a temporary ZIP file, making it efficient for automated data
    collection workflows.

    This is particularly useful for:
      * Downloading public datasets distributed as ZIP archives
      * Automating data collection pipelines that require compressed files
      * Retrieving software packages, documentation, or resources from web servers
      * Batch processing of archived data from APIs or file repositories
      * Setting up project dependencies or assets from remote sources

    The function provides visibility into archive contents through the print_contents
    option, which is helpful for validation and debugging.

    Parameters
    ----------
    url
        The URL of the ZIP file to download. Must be a valid HTTP/HTTPS URL pointing
        to a ZIP archive.
    path_to_save_folder
        The directory path where the ZIP file contents will be extracted. The directory
        will be created if it doesn't exist (when using extractall).
    unzip
        If True, extracts all files from the ZIP archive to the specified folder. If False,
        only downloads and inspects the archive without extraction. Defaults to True.
    print_contents
        If True, prints a list of all files contained in the ZIP archive to the console.
        Useful for verification and debugging. Defaults to True.

    Returns
    -------
    None
        This function extracts files to disk and optionally prints output, but does not
        return a value.

    Examples
    --------
    # Download and extract a ZIP file, printing its contents
    GetZipFile(
        url='https://example.com/data/dataset.zip',
        path_to_save_folder='./data',
        unzip=True,
        print_contents=True
    )

    # Download and inspect contents without extracting
    GetZipFile(
        url='https://example.com/archive.zip',
        path_to_save_folder='./temp',
        unzip=False,
        print_contents=True
    )

    # Silently extract without printing contents
    GetZipFile(
        url='https://example.com/resources.zip',
        path_to_save_folder='./resources',
        unzip=True,
        print_contents=False
    )

    """
    # Lazy load uncommon packages
    import zipfile
    
    # Download the zip file from the url
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Unzip the file
    if unzip:
        z.extractall(path_to_save_folder)
    
    # Print contents of the zip file
    if print_contents:
        print("Contents of the zip file:")
        for item in z.namelist():
            print(item)

