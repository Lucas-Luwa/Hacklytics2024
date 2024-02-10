import yfinance as yf
import datetime

def getOpenClosePrice(symbol, startDate, endDate):
    stock = yf.Ticker(symbol)

    startDate = datetime.datetime.strptime(startDate, "%Y-%m-%d")
    endDate = datetime.datetime.strptime(endDate, "%Y-%m-%d")

    startDate = startDate.strftime("%Y-%m-%d")
    endDate = endDate.strftime("%Y-%m-%d")

    histData = stock.history(start=startDate, end=endDate)
    openPrice = histData.iloc[0]['Open']
    closePrice = histData.iloc[-1]['Close']

    # print("Opening Price on ", startDate, ":", openPrice)
    # print("Closing Price on ", endDate, ":", closePrice)

    openPrice = round(openPrice, 2)
    closePrice = round(closePrice, 2)
    return openPrice, closePrice, startDate, endDate

#Testing
# information = getOpenClosePrice("AAPL", "2023-01-01", "2023-6-30")
# print(f"Starting Price on {information[2]}: ${information[0]}")
# print(f"Ending Price on {information[3]}: ${information[1]}")
