# Load packages
import country_converter as coco

# Define function
def AddCountryIdentifier(dataframe,
                         country_column,
                         identifier_to_add='ISO3'):
    # Select country name column
    df_countries = dataframe[[country_column]]
    df_countries = df_countries.drop_duplicates()

    # Look up identifier
    if identifier_to_add in ['ISO3', 'ISO3C', 'alpha3']:
        df_countries[identifier_to_add] = coco.CountryConverter().convert(names=df_countries[country_column],
                                                                          to='ISO3')
    elif identifier_to_add in ['ISO2', 'ISO2C', 'alpha2']:
        df_countries[identifier_to_add] = coco.CountryConverter().convert(names=df_countries[country_column],
                                                                          to='ISO2') 
    elif identifier_to_add in ['name_short', 'short name']:
        df_countries[identifier_to_add] = coco.CountryConverter().convert(names=df_countries[country_column],
                                                                          to='name_short')
    
    # Left join identifier back to original dataset
    dataframe = dataframe.merge(
        df_countries,
        how='left',
        on=[country_column]
    )
    
    # Return updated dataset
    return(dataframe)
