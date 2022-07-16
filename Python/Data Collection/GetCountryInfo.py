# Load packages
from countryinfo import CountryInfo
import numpy as np
import pycountry
import pandas as pd

# Declare function
def GetCountryInfo():
    # Get dataframe of countries
    df_countries = pd.DataFrame([country.__dict__['_fields'] for country in pycountry.countries])

    # Convert column names to title case
    df_countries.columns = df_countries.columns.str.title()

    # Rename columns
    df_countries = df_countries.rename(columns={
        'Alpha_2': 'ISO2C',
        'Alpha_3': 'ISO3C',
        'Numeric': 'Numeric Code'
    })

    # Replace underscore in column names
    df_countries.columns = df_countries.columns.str.replace("_", " ",
                                                            regex=False)

    # Drop flag column
    df_countries = df_countries.drop(columns=['Flag'])

    # Create placeholder dataframe for additional country information
    df_temp = pd.DataFrame()

    # Iterate through each country in dataframe
    for i in range(0, len(df_countries.index)):
        # Search for current country
        try:
            df_country_info = CountryInfo(df_countries['Name'][i])
            df_country_info = pd.json_normalize(df_country_info.info())
        except:
            try:
                df_country_info = CountryInfo(df_countries['Official Name'][i])
                df_country_info = pd.json_normalize(df_country_info.info())
            except:
                try:
                    df_country_info = CountryInfo(df_countries['Common Name'][i])
                    df_country_info = pd.json_normalize(df_country_info.info())
                except:
                    continue
        
        # Append to placeholder
        df_temp = pd.concat([df_temp, df_country_info])
        
    # Select necessary columns
    df_temp = df_temp[[
        'ISO.alpha3',
        'altSpellings',
        'borders',
        'capital',
        'capital_latlng',
        'currencies',
        'demonym',
        'flag',
        'languages',
        'latlng',
        'nativeName',
        'provinces',
        'region',
        'subregion',
        'timezones',
        'wiki'
    ]]

    # Convert column names to title case
    df_temp.columns = df_temp.columns.str.title()

    # Fix column names
    df_temp = df_temp.rename(columns={
        'Iso.Alpha3': 'ISO3C',
        'Altspellings': 'Alternate Spellings',
        'Capital': 'Capital City',
        'Capital_Latlng': 'Capital City Center Coordinates',
        'Latlng': 'Country Center Coordinates',
        'Nativename': 'Native Name'
    })

    # Left join additional country info
    df_countries = df_countries.merge(
        df_temp,
        how='left',
        on=['ISO3C']
    )

    # Replace blanks with nan
    df_countries = df_countries.replace("", np.nan)
    for col in df_countries.columns:
        df_countries[col] = np.where(
            df_countries[col].str.len() == 0,
            np.nan,
            df_countries[col]
        )

    # Return dataframe
    return(df_countries)
