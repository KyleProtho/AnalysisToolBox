# Load packages
import matplotlib.pyplot as plt 

# Declare function
def FetchUSShapefile(state=None,
                     county=None,
                     geography='census tract',
                     census_year=2021):
    """
    Fetch a geographical shapefile from the U.S. Census Bureau's TIGER database.

    This function retrieves geographical boundary data (shapefiles) from the U.S. Census Bureau's
    TIGER (Topologically Integrated Geographic Encoding and Referencing) database. The data is
    returned as a GeoDataFrame, ready for spatial analysis and visualization. This provides an
    easy interface to download official U.S. Census geographic boundaries without manual file
    management or data conversion.

    TIGER shapefiles are the authoritative source for U.S. Census geographic boundaries and
    are essential for:
      * Demographic and socioeconomic spatial analysis
      * Electoral district mapping and redistricting analysis
      * Public health surveillance and healthcare resource planning
      * Crime mapping and law enforcement jurisdiction analysis
      * Educational planning and school district analysis
      * Market research and business location intelligence

    The function supports multiple geographic levels (from nation-wide down to census blocks)
    and allows filtering by state and county when applicable. The underlying pygris library
    handles caching to improve performance on repeated requests.

    Parameters
    ----------
    state
        Two-letter state abbreviation or full state name for which to fetch the shapefile.
        Required for most geography types (e.g., counties, tracts) but not for nation-level
        geographies. Defaults to None.
    county
        County name for which to fetch the shapefile. Required for fine-grained geographies
        like census tracts, block groups, and blocks. Ignored for broader geographies.
        Defaults to None.
    geography
        Type of geographical unit to fetch. Must be one of: 'nation', 'divisions', 'regions',
        'states', 'counties', 'zipcodes', 'tract', 'block groups', 'blocks', 'school districts',
        'county subdivisions', 'congressional districts', 'state legislative districts',
        'voting districts'. Defaults to 'census tract'.
    census_year
        Census year for which to fetch the shapefile. Different years may have different
        boundaries due to redistricting or geographic changes. Defaults to 2021.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the requested geographic boundaries with associated
        attributes (FIPS codes, names, etc.) and geometry information.

    Examples
    --------
    # Fetch all census tracts for a specific state
    tracts_gdf = FetchUSShapefile(
        state='CA',
        geography='tract',
        census_year=2021
    )

    # Fetch all counties in the United States
    counties_gdf = FetchUSShapefile(
        geography='counties',
        census_year=2020
    )

    # Fetch census block groups for a specific county
    block_groups_gdf = FetchUSShapefile(
        state='TX',
        county='Harris',
        geography='block groups',
        census_year=2021
    )

    # Fetch ZIP code tabulation areas (ZCTAs) for a state
    zipcodes_gdf = FetchUSShapefile(
        state='NY',
        geography='zipcodes'
    )

    """
    # Lazy load uncommon packages
    import folium 
    import geopandas as gp
    import mapclassify
    import pygris
    
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

