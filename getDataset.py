import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def options_chain(symbol):

    tk = yf.Ticker(symbol)
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
    print(tk.history('max'))
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

def options_history(symbol, interval="60m", start="2022-02-11", end="2024-02-09"):

    tk = yf.Ticker(symbol)
    return tk.history(interval=interval, start=start, end=end)

def export_pandas(dataframe, path, type):
    if type == "pickle":
        dataframe.to_pickle(path+ ".pkl")
    else:
        dataframe.to_csv(path+ ".csv")


def generate_random_symbols(percentage, nysedf, nasdaqdf):
    nyseRows = nysedf.shape[0]
    nasdaqRows = nasdaqdf.shape[0]

    #Pick out 2.5% of Tick symbols from NYSE and NASDAQ
    randnyseRows = nysedf.sample(n=int(nyseRows*percentage))
    randNasdaqRows = nysedf.sample(n=int(nasdaqRows*percentage))

    randnyseRows = randnyseRows.sort_index()
    randNasdaqRows = randNasdaqRows.sort_index()

    nyseTick = randnyseRows.iloc[:, 0]
    nasdaqTick = randNasdaqRows.iloc[:,0]
    nyseNames = randnyseRows.iloc[:, 1]
    nasdaqNames = randNasdaqRows.iloc[:,1]

    return nyseTick, nasdaqTick, nyseNames, nasdaqNames

# export_pandas(options_history("AAPL", "60m", "2022-02-10", "2024-02-09"), "./apple_stock_options.csv")

def generate_datasets(percentage):
    nyse = 'Tickers/NYSELarge+Ticker.xlsx'
    nasdaq = 'Tickers/NASDAQLarge+Ticker.xlsx'
    
    nysedf = pd.read_excel(nyse)
    nasdaqdf = pd.read_excel(nasdaq)

    nyse_symbol_list = []
    nyse_name_list = []
    nyse_date_list = []
    nyse_open_list = []
    nyse_high_list = []
    nyse_low_list = []
    nyse_close_list = []
    nyse_volume_list = []
    nyse_dividends_list = []
    nyse_stock_splits_list = []

    nasdaq_symbol_list = []
    nasdaq_name_list = []
    nasdaq_date_list = []
    nasdaq_open_list = []
    nasdaq_high_list = []
    nasdaq_low_list = []
    nasdaq_close_list = []
    nasdaq_volume_list = []
    nasdaq_dividends_list = []
    nasdaq_stock_splits_list = []

    nyseTick, nasdaqTick, nyseNames, nasdaqNames = generate_random_symbols(percentage, nysedf, nasdaqdf)
    for i in range(len(nyseTick)):
        print(options_history(nyseTick.iloc[i]))

def options_histOpt(symbol):
    ticker_symbol = symbol
    expiration_date = '2024-02-16'

    options_data = yf.Ticker(ticker_symbol).option_chain(expiration_date)

    call_options = options_data.calls
    put_options = options_data.puts

    print("Call Options:")
    print(call_options.head())

    print("\nPut Options:")
    print(put_options.head())

# print(options_history("RIO", "1d", "2024-01-09", "2024-02-09"))
# print(options_chain("GOOG"))
# print(options_histOpt("GOOG"))
# options_chain("TSLA")

