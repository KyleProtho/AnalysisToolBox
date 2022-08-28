# Load packages
import pandas as pd
import networkx as nx

# Declare function (be sure to run the GetCountryInfo function first)
def CreateCountryBorderNetwork(dataframe_country_info):
    # Drop unnecessary columns
    df_borders = dataframe_country_info[[
        'ISO3C',
        'Borders'
    ]]
    
    # Separate borders into separate rows
    df_borders = df_borders.explode('Borders')
    
    # Create network of borders
    G = nx.from_pandas_edgelist(
        df_borders,
        source='ISO3C',
        target='Borders',
        create_using=nx.Graph()
    )
    
    # Get latitude and longitude
    df_coords = dataframe_country_info[[
        'ISO3C',
        'Name',
        'Country Center Coordinates'
    ]].dropna()
    
    # Split latitude and longitude
    df_coords[['Latitude', 'Longitude']] = pd.DataFrame(
        df_coords['Country Center Coordinates'].tolist(),
        index=df_coords.index
    )
    
    # Drop center coords column
    df_coords = df_coords.drop(columns=['Country Center Coordinates'])
    
    # Add name & lat.long to nodes in network object
    df_coords = df_coords.set_index('ISO3C').to_dict('index')
    nx.set_node_attributes(G, df_coords)
    
    # Return network object
    return(G)
