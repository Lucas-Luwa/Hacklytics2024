import requests
from startEndPrices import getOpenClosePrice
from getDataset import options_history
from optPercentReturn import calcPercentReturn

def get_full_dataset():
  endpoint = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
  query = "SELECT * FROM `option_chain` WHERE `date` >= '2023-01-01' AND `expiration` < '2024-02-10' ORDER BY `date` ASC"
  res = requests.get(endpoint, params={'q': query})
  data = res.json()['rows']

  # {'date': '2019-02-23', 'act_symbol': 'SBAC', 'expiration': '2019-04-18', 'strike': '200.00', 
  # 'call_put': 'Call', 'bid': '0.20', 'ask': '0.65', 'vol': '0.1687', 'delta': '0.0940', 
  # 'gamma': '0.0142', 'theta': '-0.0196', 'vega': '0.1177', 'rho': '0.0246'}

  dataset = []

  for row in data:
    try:
      info = getOpenClosePrice(row['act_symbol'], row['date'], row['expiration'])
      entry = row
      entry['open_price'] = info[0]
      entry['close_price'] = info[1]
      entry['mark'] = (float(row['bid']) + float(row['ask'])) / 2
      history = options_history(row['act_symbol'], "60m", row['date'], row['expiration'])
      entry['history'] = history["Open"].tolist()
      entry["percent_return"] = calcPercentReturn(entry['call_put'], float(entry['strike']), entry['close_price'], entry['mark'])
      dataset.append(entry)
    except:
      print("exception")

  return dataset

# {'date': '2023-01-02', 'act_symbol': 'ABG', 'expiration': '2023-01-20', 'strike': '210.00', 
#  'call_put': 'Put', 'bid': '28.70', 'ask': '32.70', 'vol': '0.3293', 'delta': '-0.9972', 
#  'gamma': '0.0033', 'theta': '0.0000', 'vega': '0.0052', 'rho': '0.0000', 
#  'open_price': 180.06, 'close_price': 184.04, 'mark': 30.700000000000003, 
#  'history': [180.05999755859375, 178.44000244140625, 177.50999450683594, ..., 183.11749267578125, 183.38999938964844], 
#  'percent_return': -15.44}
