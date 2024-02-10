import pandas as pd

nyse = 'Tickers/NYSELarge+Ticker.xlsx'
nasdaq = 'Tickers/NASDAQLarge+Ticker.xlsx'

nysedf = pd.read_excel(nyse)
nasdaqdf = pd.read_excel(nasdaq)

nyseRows = nysedf.shape[0]
nasdaqRows = nasdaqdf.shape[0]

#Pick out 2.5% of Tick symbols from NYSE and NASDAQ
randnyseRows = nysedf.sample(n=int(nyseRows*0.025))
randNasdaqRows = nysedf.sample(n=int(nasdaqRows*0.025))

randnyseRows = randnyseRows.sort_index()
randNasdaqRows = randNasdaqRows.sort_index()

nyseTick = randnyseRows.iloc[:, 0]
nasdaqTick = randNasdaqRows.iloc[:,0]

print(nyseTick)
print(nasdaqTick)