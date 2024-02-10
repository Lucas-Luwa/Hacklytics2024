import math

def sigmoid(x):
    return float(1/(1 + math.exp(-x)))

def calcPercentReturn(type, strike, closingPrice, premium):
    if type == 'Call':
        rawReturn = closingPrice - (strike + premium)
    elif type == 'Put':
        rawReturn = (strike - premium) - closingPrice
    else:
        return ValueError("Must use 'Call' or 'Put' option type")

    percent = round(float(rawReturn/premium) * 100, 2)
    return percent

#Testing
# print(calcPercentReturn('Call', 2, 5, 1))
