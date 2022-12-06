from datetime import datetime
import numpy as np
import pandas as pd
import wbgapi as wb

# Function that downloads socioeconomic data from the World Bank for all countries and returns a dataframe 
def GetWorldBankDataByCountry(dict_indicators={
        # Population indicators
        "SP.POP.TOTL": "Population, total",
        "SP.POP.GROW": "Population growth (annual %)",
        "SP.POP.DPND": "Population, ages 15-64, dependency ratio (% of working-age population)",
        "SP.POP.DPND.OL": "Population ages 65 and above, old age dependency ratio (% of working-age population)",
        "SP.POP.DPND.YG": "Population ages 0-14, youth dependency ratio (% of working-age population)",
        "SP.POP.0014.TO.ZS": "Population ages 0-14 (% of total)",
        "SP.POP.1564.TO.ZS": "Population ages 15-64 (% of total)",
        "SP.POP.65UP.TO.ZS": "Population ages 65 and above (% of total)",
        "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
        "SP.URB.GROW": "Urban population growth (annual %)",
        "SP.RUR.TOTL.IN.ZS": "Rural population (% of total population)",
        "SP.RUR.GROW": "Rural population growth (annual %)",
        "SP.DYN.CBRT.IN": "Crude birth rate (per 1,000 people)",
        # Economic indicators
        "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
        "NY.GDP.PCAP.KD.ZG": "GDP per capita growth (annual %)",
        "NY.GNP.PCAP.CD": "GNI per capita, Atlas method (current US$)",
        "NY.GNP.MKTP.CD": "GNI, Atlas method (current US$)",
        "NY.GNP.MKTP.KD.ZG": "GNI growth (annual %)",
        "NY.GNP.PCAP.KD.ZG": "GNI per capita growth (annual %)",
        # Health indicators
        "HD.HCI.OVRL": "Health care index (HCI)",
        "HD.HCI.MORT": "Child mortality rate, under-5 (per 1,000 live births)",
        "HD.HCI.AMRT": "Survival rate, ages 15 to 65",
        # Technology indicators
        "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
        "IT.CEL.SETS.P2": "Mobile cellular subscriptions (per 100 people)",
        # Ease of doing business indicators
        "IC.BUS.EASE.XQ": "Ease of doing business index (1=most business-friendly regulations)",
        "IC.REG.COST.PC.ZS": "Cost to start a business (% of GNI per capita)",
        "IC.REG.PROC": "Time required to start a business (days)",
        # World Bank governance indicators
        "CC.EST": "Control of corruption: Estimate",
        "GE.EST": "Government effectiveness: Estimate",
        "PV.EST": "Political stability and absence",
        "PC.EST": "Public trust in politicians: Estimate",
        "RL.EST": "Rule of law: Estimate",
        "RQ.EST": "Regulatory quality: Estimate",
        "VA.EST": "Voice and accountability: Estimate"
    },
                              end_year=datetime.now().year,
                              start_year=datetime.now().year - 7,
                              indicators_as_columns=True):
    
    # Fetch socioeconomic indicators for all countries
    data_countries = wb.data.DataFrame(
        series=dict_indicators.keys(), 
        economy="all", 
        time=range(start_year, end_year),
        skipBlanks=True,
        columns="series"
    )
    
    # Rename the columns
    data_countries = data_countries.rename(columns=dict_indicators)
    
    # Reset the index and rename the country and year columns
    data_countries = data_countries.reset_index().rename(columns={
        "economy": "ISO3C", 
        "time": "Year"
    })
    
    # Clean the year column
    data_countries["Year"] = data_countries["Year"].str.replace("YR", "").astype(int)
    
    # Return the dataframe
    return(data_countries)

# # Test the function
# data_countries = GetWorldBankDataByCountry()
