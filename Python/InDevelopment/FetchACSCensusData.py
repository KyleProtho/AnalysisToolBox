# # Load packages
# import folium 
# import geopandas as gp
# import mapclassify
# import matplotlib.pyplot as plt 
# import pygris
# from pygris.data import get_census

# # Declare function
# def FetchACSCensusData(list_of_variables,
#                        acs_dataset="acs/acs5",
#                        state=None,
#                        county=None,
#                        geography='census tract',
#                        census_year=2021):
    
#     # Ensure that geography is a valid option
#     if geography not in ['divisions', 'regions', 'states', 'counties', 'zipcodes', 'tract', 'block groups', 'blocks', 'school districts', 'county subdivisions', 'congressional districts', 'state legislative districts', 'voting districts']:
#         raise ValueError('geography must be one of the following: divisions, regions, states, counties, zipcodes, tract, block groups, blocks, school districts, county subdivisions, congressional districts, state legislative districts, voting districts')
    
#     # Create API call parameters dictionary
#     api_call_params = {
#         "time": census_year
#     }
    
#     # Generate parameters for API call based on state
#     if geography in ['divisions', 'regions', 'states']:
#         pass
#     else:
#         if state is None:
#             api_call_params["in"] = "state:*"
#         else:
#             state_fips = pygris.states(year=census_year).loc[pygris.states(year=census_year)['NAME']==state, 'STATEFP'].values[0]
#             api_call_params["in"] = f"state:{state_fips}"
            
#     # Generate parameters for API call based on county
#     if geography in ['divisions', 'regions', 'states', 'counties']:
#         pass
#     else:
#         if county is None:
#             api_call_params["for"]="county:*"
#         else:
#             county_fips = pygris.counties(state=state, year=census_year).loc[pygris.counties(state=state, year=census_year)['NAME']==county, 'COUNTYFP'].values[0]
#             api_call_params["for"] = f"county:{county_fips}"
    
#     # Generate parameters for API call based on geographies
#     if geography=='divisions':
#         list_of_variables.append("DIVISION")
#     elif geography=='regions':
#         list_of_variables.append("REGION")
#     elif geography=='states':
#         list_of_variables.append("STATE")
#     elif geography=='counties':
#         list_of_variables.append("COUNTY")
#     elif geography=='zipcodes':
#         list_of_variables.append("ZCTA")
#     elif geography=='tract':
#         list_of_variables.append("TRACT")
#     elif geography=='block groups':
#         list_of_variables.append("BLKGRP")
#     elif geography=='blocks':
#         list_of_variables.append("BLOCK")
#     elif geography=='school districts':
#         list_of_variables.append("SDUNI")
#     elif geography=='county subdivisions':
#         list_of_variables.append("COUSUB")
#     elif geography=='congressional districts':
#         list_of_variables.append("CD116")
#     elif geography=='state legislative districts':
#         list_of_variables.append("SLDU")
#     elif geography=='voting districts':
#         list_of_variables.append("VTD")
    
#     # Fetch data from U.S. Census Bureau. Visit https://www.census.gov/data/developers/data-sets.html for API information.
#     data_census = get_census(
#         dataset=acs_dataset,
#         variables=list_of_variables,
#         params=api_call_params,
#         return_geoid=True,
#         guess_dtypes=True
#     )
    
#     # Return data
#     return(data_census)


# # Test function
# pa_county_subdivision_data = FetchACSCensusData(
#     state='Pennsylvania',
#     county='Allegheny',
#     geography='county subdivisions',
#     list_of_variables=["B01001_001E"],
#     acs_dataset="acs/acs5/subject"
# )
