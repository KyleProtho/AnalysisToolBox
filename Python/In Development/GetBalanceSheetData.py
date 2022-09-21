# Load packages
import pandas as pd
import numpy as np

# Set arguments
stock_ticker = "AAPL"

# Set Market Watch URL
data_url = "https://www.marketwatch.com/investing/stock/" + stock_ticker + "/financials/balance-sheet"


### ASSETS ###

# Read balance sheet Asset table
df_balance_sheet = pd.read_html(data_url,
                                attrs={'aria-label': 'Financials - Assets data table'})
df_balance_sheet = df_balance_sheet[0]

# Drop repetitive words from 'Item' column
df_balance_sheet['Item'] =  df_balance_sheet['Item  Item'].str.partition('  ')[0]
df_balance_sheet = df_balance_sheet[ ['Item'] + [ col for col in df_balance_sheet.columns if col != 'Item' ] ]

# Drop trend and item item columns
df_balance_sheet = df_balance_sheet.drop(columns=[
    '5-year trend',
    'Item  Item'
])

# Filter out growth rows
df_balance_sheet = df_balance_sheet[
    ~df_balance_sheet['Item'].str.contains('Growth')
]

# Add item category column
df_balance_sheet['Item Category'] = "Asset"

# Convert to long form
df_balance_sheet = pd.melt(
    df_balance_sheet, 
    id_vars=['Item', 'Item Category'], 
    var_name='Year',
    value_name='Value'
)

# Replace '-' with NaN
df_balance_sheet['Value'] = np.where(
    df_balance_sheet['Value'].str.contains('-'), 
    np.nan, 
    df_balance_sheet['Value']
)

# Add value scale column 
df_balance_sheet['Value Unit of Measure'] = np.where(
    df_balance_sheet['Value'].isna(), np.nan,
    np.where(df_balance_sheet['Value'].str.contains('%', regex=False), 'Percent',
    np.where(df_balance_sheet['Value'].str.contains('M', regex=False), 'Dollars',
    np.where(df_balance_sheet['Value'].str.contains('B', regex=False), 'Dollars', 'Ratio'
))))

# Replace M and B with appropriate number of zeros
df_balance_sheet['Value - revised'] = np.where(
    df_balance_sheet['Value'].str.contains('%', regex=False), df_balance_sheet['Value'].str.replace('%', ''),
    np.where(df_balance_sheet['Value'].str.contains('M', regex=False), df_balance_sheet['Value'].str.replace('M', ''),
    np.where(df_balance_sheet['Value'].str.contains('B', regex=False), df_balance_sheet['Value'].str.replace('B', ''), df_balance_sheet['Value']
)))
df_balance_sheet['Value - revised'] = df_balance_sheet['Value - revised'].str.replace("(", "-", regex=False)
df_balance_sheet['Value - revised'] = df_balance_sheet['Value - revised'].str.replace(")", "", regex=False)
df_balance_sheet['Value - revised'] = pd.to_numeric(df_balance_sheet['Value - revised'])
df_balance_sheet['Value - revised'] = np.where(
    df_balance_sheet['Value'].str.contains('%', regex=False), df_balance_sheet['Value - revised'] / 100,
    np.where(df_balance_sheet['Value'].str.contains('M', regex=False), df_balance_sheet['Value - revised'] * 1000000,
    np.where(df_balance_sheet['Value'].str.contains('B', regex=False), df_balance_sheet['Value - revised'] * 1000000000, df_balance_sheet['Value - revised']
)))
df_balance_sheet['Value'] = df_balance_sheet['Value - revised']
df_balance_sheet = df_balance_sheet.drop(columns=[
    'Value - revised'
])


### LIABILITIES & SHAREHOLDERS' EQUITY ###
