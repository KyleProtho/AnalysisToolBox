# Load packages
import pandas as pd
import numpy as np

# Declare function
def GeocodeUSAddresses(dataframe,
                       address_column_name,
                       latitude_column_name='Latitude',
                       longitude_column_name='Longitude'):
    """Geocodes addresses in a dataframe using U.S. Census Bureau's geocoding service.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the addresses to be geocoded.
        address_column_name (str): The name of the column in the dataframe that contains the addresses.
        latitude_column_name (str, optional): The name of the column to be created in the dataframe to store the latitude. Defaults to 'Latitude'.
        longitude_column_name (str, optional): The name of the column to be created in the dataframe to store the longitude. Defaults to 'Longitude'.

    Returns:
        pandas.DataFrame: The dataframe with additional columns for latitude and longitude.
    """
    # Lazy load uncommon packages
    import censusgeocode as cg
    
    # Define a function to handle geocoding for a single address
    def get_geocode(address):
        try:
            # Using 'street' as the search parameter allows for freeform address input
            result = cg.onelineaddress(address, returntype='locations')
            # The API returns a list of matches; we'll take the first (best) match if any
            if result and len(result) > 0:
                top_result = result[0]
                return top_result['coordinates']['x'], top_result['coordinates']['y'] # x = longitude, y = latitude
        except Exception as e:
            print(f"Error geocoding address {address}: {str(e)}")
            return np.nan, np.nan
        
        return np.nan, np.nan
    
    # Apply the function to the desired column and create new columns for latitude and longitude
    dataframe[latitude_column_name], dataframe[longitude_column_name] = zip(*dataframe[address_column_name].apply(get_geocode))
    
    # Return the dataframe with the new columns
    return dataframe
