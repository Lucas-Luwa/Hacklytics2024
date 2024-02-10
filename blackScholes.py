from scipy.stats import norm
import math
from datetime import datetime
import yfinance as yf

# Inputs
#type -> Call or Put  
#S -> Current stock price
#K -> Strike price
#t ->  Time to expiration in years (Alternatively, provide 2 dates)
#r -> Risk-free interest rate
#sigma -> Volatility

def blackScholes(optType, S, K, time1, time2, r, sigma, flag):
    if flag == 0:
        t = time1 # If flag is set to 0, we're good to go. 
    if flag == 1:
        date1 = datetime.strptime(time1, '%Y-%m-%d')
        date2 = datetime.strptime(time2, '%Y-%m-%d')
        dayDiff = (date2 - date1).days
        t = float(dayDiff/365)
        # print(t)

    #d1 Calculation
    currPriceOverStrike = math.log(float(S/K))
    interestAndTime = (r + float((sigma ** 2)/2) ) * t
    volRootTime = sigma * math.sqrt(t)
    d1 = (currPriceOverStrike + interestAndTime) / volRootTime

    #d2 Calculation
    d2 = d1 - sigma * math.sqrt(t)

    if optType == 'Call':
        premium = norm.cdf(d1) * S - norm.cdf(d2) * K * math.exp(-r * t)
    elif optType == 'Put':
        premium = K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Must use 'Call' or 'Put' option type")

    return premium

# Inputs
#type -> Call or Put  
#S -> Current stock price
#K -> Strike price
#t ->  Time to expiration in years (Alternatively, provide 2 dates)
#r -> Risk-free interest rate
#sigma -> Volatility

#Testing
type = 'Call'
#With a decimal value
optPrem = round(blackScholes(type, 180.06, 210, 0.41, 0, 0.03, 0.3, 0), 3)
#Given two dates
optPrem2 = round(blackScholes(type, 150.86, 105, '2023-01-02', '2023-01-20',  0.03, .3132,  1), 3)

print(f"{type} option premium: {optPrem}")
print(f"{type} option premium: {optPrem2}")