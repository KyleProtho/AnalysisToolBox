# Load packages
import folium 
import mapclassify
import matplotlib.pyplot as plt 
import geopandas as gp
import pygris

# Define function to fetch shapefile
def FetchUSShapefile(state=None,
                     county=None,
                     geography='census tract',
                     census_year=2021):
    # Ensure that geography is a valid option
    if geography not in ['nation', 'divisions', 'regions', 'states', 'counties', 'zipcodes', 'tract', 'block groups', 'blocks', 'school districts', 'county subdivisions', 'congressional districts', 'state legislative districts', 'voting districts']:
        raise ValueError('geography must be one of the following: nation, divisions, regions, states, counties, zipcodes, tract, block groups, blocks, school districts, county subdivisions, congressional districts, state legislative districts, voting districts')
    
    # Fetch shapefile from TIGER database from U.S. Census Bureau
    if geography=='nation':
        shapefile = pygris.nation(year=census_year)
    elif geography=='divisions':
        shapefile = pygris.divisions(year=census_year)
    elif geography=='regions':
        shapefile = pygris.regions(year=census_year)
    elif geography=='states':
        shapefile = pygris.states(year=census_year)
    elif geography=='counties':
        shapefile = pygris.counties(state=state, 
                                    year=census_year)
    elif geography=='zipcodes':
        shapefile = pygris.zctas(state=state,
                                 year=census_year)
    elif geography=='tract':
        shapefile = pygris.tracts(state=state, 
                                  county=county, 
                                  year=census_year)
    elif geography=='block groups':
        shapefile = pygris.block_groups(state=state, 
                                        county=county, 
                                        year=census_year)
    elif geography=='blocks':
        shapefile = pygris.blocks(state=state, 
                                  county=county, 
                                  year=census_year)
    elif geography=='school districts':
        shapefile = pygris.school_districts(state=state,
                                            county=county, 
                                            year=census_year)
    elif geography=='county subdivisions':
        shapefile = pygris.county_subdivisions(state=state,
                                               county=county,
                                               year=census_year)
    elif geography=='congressional districts':
        shapefile = pygris.congressional_districts(state=state,
                                                   county=county,
                                                   year=census_year)
    elif geography=='state legislative districts':
        shapefile = pygris.state_legislative_districts(state=state,
                                                       county=county,
                                                       year=census_year)
    elif geography=='voting districts':
        shapefile = pygris.voting_districts(state=state,
                                            county=county,
                                            year=census_year)
    
    # Return the shapefile
    return(shapefile)


# # Test function
# pa_counties = FetchUSShapefile(
#     state='Pennsylvania',
#     geography='counties',
#     census_year=2021
# )
# pa_counties.explore()

# pa_county_subdivisions = FetchUSShapefile(
#     state='Pennsylvania',
#     county='Allegheny',
#     geography='county subdivisions',
# )
# pa_county_subdivisions.explore()
