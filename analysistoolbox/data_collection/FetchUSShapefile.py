# Load packages
import folium 
import geopandas as gp
import mapclassify
import matplotlib.pyplot as plt 
import pygris

# Declare function
def FetchUSShapefile(state=None,
                     county=None,
                     geography='census tract',
                     census_year=2021):
    """
    Fetches a geographical shapefile from the TIGER database of the U.S. Census Bureau.

    Parameters:
    state (str, optional): The state for which the shapefile is to be fetched. Default is None.
    county (str, optional): The county for which the shapefile is to be fetched. Default is None.
    geography (str, optional): The type of geographical unit for which the shapefile is to be fetched. 
                               Must be one of the following: 'nation', 'divisions', 'regions', 'states', 
                               'counties', 'zipcodes', 'tract', 'block groups', 'blocks', 'school districts', 
                               'county subdivisions', 'congressional districts', 'state legislative districts', 
                               'voting districts'. Default is 'census tract'.
    census_year (int, optional): The census year for which the shapefile is to be fetched. Default is 2021.

    Returns:
    geopandas.GeoDataFrame: The fetched shapefile as a GeoDataFrame.
    """
    
    # Ensure that geography is a valid option
    if geography not in ['nation', 'divisions', 'regions', 'states', 'counties', 'zipcodes', 'tract', 'block groups', 'blocks', 'school districts', 'county subdivisions', 'congressional districts', 'state legislative districts', 'voting districts']:
        raise ValueError('geography must be one of the following: nation, divisions, regions, states, counties, zipcodes, tract, block groups, blocks, school districts, county subdivisions, congressional districts, state legislative districts, voting districts')
    
    # Ensure that geography is a valid option
    valid_geographies = {
        'nation': pygris.nation,
        'divisions': pygris.divisions,
        'regions': pygris.regions,
        'states': pygris.states,
        'counties': pygris.counties,
        'zipcodes': pygris.zctas,
        'tract': pygris.tracts,
        'block groups': pygris.block_groups,
        'blocks': pygris.blocks,
        'school districts': pygris.school_districts,
        'county subdivisions': pygris.county_subdivisions,
        'congressional districts': pygris.congressional_districts,
        'state legislative districts': pygris.state_legislative_districts,
        'voting districts': pygris.voting_districts
    }

    if geography not in valid_geographies:
        raise ValueError('geography must be one of the following: ' + ', '.join(valid_geographies.keys()))
    
    # Fetch shapefile from TIGER database from U.S. Census Bureau
    shapefile_function = valid_geographies[geography]
    if geography in ['nation', 'divisions', 'regions', 'states']:
        shapefile = shapefile_function(year=census_year)
    elif geography in ['counties', 'zipcodes']:
        shapefile = shapefile_function(state=state, year=census_year)
    else:
        shapefile = shapefile_function(state=state, county=county, year=census_year)
    
    # Return the shapefile
    return(shapefile)

