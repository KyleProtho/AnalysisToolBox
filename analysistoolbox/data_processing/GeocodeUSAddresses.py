# Load packages
import pandas as pd
import numpy as np

# Declare function
def GeocodeUSAddresses(dataframe,
                       address_column_name,
                       latitude_column_name='Latitude',
                       longitude_column_name='Longitude'):
    """
    Geocode U.S. addresses into latitude and longitude coordinates using the U.S. Census Bureau service.

    This function provides programmatic access to the U.S. Census Bureau's Geocoding Services 
    API to convert human-readable U.S. addresses into geographic coordinates. It utilizes 
    the `censusgeocode` wrapper to perform one-line address lookups, which are effective 
    for freeform address strings including street, city, state, and ZIP code information.

    The function is particularly useful for:
      * Creating geospatial visualizations (e.g., heatmaps or point maps)
      * Performing proximity analysis (calculating distances between points)
      * Regional market segmentation based on physical location
      * Enhancing customer data with geographic coordinates for logistics
      * Preparing data for spatial joins with census tracts or block groups
      * Validating and standardizing address information for U.S. locations

    The U.S. Census Bureau service is free to use but is optimized specifically for 
    United States addresses. If the service fails to find a location or encounters 
    an error, the function returns NaN for the coordinates to ensure the data 
    pipeline continues without interruption.

    Parameters
    ----------
    dataframe
        The pandas DataFrame containing the U.S. addresses to be geocoded.
    address_column_name
        The name of the column in the DataFrame that contains the full address 
        strings (e.g., '1600 Amphitheatre Parkway, Mountain View, CA 94043').
    latitude_column_name
        The name of the new column to be created for storing latitude values. 
        Defaults to 'Latitude'.
    longitude_column_name
        The name of the new column to be created for storing longitude values. 
        Defaults to 'Longitude'.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two additional columns for latitude and longitude. 
        Unmatched addresses will result in NaN values in these columns.

    Examples
    --------
    # Geocode a list of corporate headquarters
    import pandas as pd
    companies = pd.DataFrame({
        'name': ['Google', 'Apple', 'Microsoft'],
        'address': [
            '1600 Amphitheatre Pkwy, Mountain View, CA 94043',
            'One Apple Park Way, Cupertino, CA 95014',
            'One Microsoft Way, Redmond, WA 98052'
        ]
    })
    companies = GeocodeUSAddresses(companies, 'address')
    # Adds 'Latitude' and 'Longitude' columns to the DataFrame

    # Geocode using custom column names for spatial analysis
    stores = pd.DataFrame({
        'store_id': [101, 102],
        'full_str': ['350 5th Ave, New York, NY 10118', 'Invalid Address']
    })
    stores = GeocodeUSAddresses(
        stores, 
        'full_str', 
        latitude_column_name='lat', 
        longitude_column_name='lon'
    )
    # Match for Empire State Building found; 'Invalid Address' results in NaN.

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
