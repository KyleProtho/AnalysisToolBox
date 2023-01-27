import io
import requests
import zipfile

def GetZipFile(url,
               path_to_save_folder,
               unzip=True,
               print_contents=True):
    """_summary_
    This function downloads a zip file from a url and saves it to a specified folder. It can also unzip the file and print the contents of the zip file.
    
    Args:
        url (str): The url of the zip file to download.
        path_to_save_folder (str): The path to the folder where the zip file will be saved.
        unzip (bool, optional): Whether or not to unzip the file. Defaults to True.
        print_contents (bool, optional): Whether or not to print the contents of the zip file. Defaults to True.
    """
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
