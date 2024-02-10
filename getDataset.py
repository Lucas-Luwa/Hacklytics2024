import pandas as pd
import numpy as np
import yfinance as yf
import datetime

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options

    # Initialize an empty list to hold individual option DataFrames
    options_list = []

    # Get options for each expiration
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.concat([opt.calls, opt.puts])
        opt['expirationDate'] = e
        options_list.append(opt)

    # Concatenate all option DataFrames into a single DataFrame
    options = pd.concat(options_list, ignore_index=True)

    # Bizarre error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days=1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns=['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options

def options_history(symbol, interval, start, end):

    tk = yf.Ticker(symbol)
    return tk.history(interval=interval, start=start, end=end)

print(options_history("RIO", "1d", "2024-01-09", "2024-02-09"))
