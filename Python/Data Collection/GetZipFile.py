import io
import requests
import zipfile

def GetZipFile(url,
               path_to_save_folder,
               unzip=True,
               print_contents=True):
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
