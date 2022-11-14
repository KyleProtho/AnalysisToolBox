import numpy as np
import pandas as pd
import yfinance as yf

# A function that gets balance sheet data of a company from Yahoo Finance
def GetBalanceSheetData(ticker):
    # Get balance sheet data from Yahoo Finance
    data_balance_sheet = yf.Ticker(ticker).balance_sheet
    
    # Convert to long form
    data_balance_sheet = data_balance_sheet.reset_index()
    data_balance_sheet = data_balance_sheet.melt(
        id_vars=['index'], 
        var_name='Year',
        value_name='Value'
    )
    
    # Rename 'index' column to 'Item'
    data_balance_sheet = data_balance_sheet.rename(columns={'index': 'Item'})
    
    # Return balance sheet data
    return data_balance_sheet

# Test function
data_apple = GetBalanceSheetData('AAPL')
