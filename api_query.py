import requests
from startEndPrices import getOpenClosePrice

def get_full_dataset():
  endpoint = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master?q=SELECT+*+FROM+%60option_chain%60"
  query = "SELECT * FROM `option_chain` ORDER BY `date` ASC"
  res = requests.get(endpoint)
  data = res.json()['rows']

  # {'date': '2019-02-23', 'act_symbol': 'SBAC', 'expiration': '2019-04-18', 'strike': '200.00', 
  # 'call_put': 'Call', 'bid': '0.20', 'ask': '0.65', 'vol': '0.1687', 'delta': '0.0940', 
  # 'gamma': '0.0142', 'theta': '-0.0196', 'vega': '0.1177', 'rho': '0.0246'}

  dataset = []

  for row in data:
    try:
      info = getOpenClosePrice(row['act_symbol'], row['date'], row['expiration'])
      entry = row
      print(info)
      entry['open_price'] = info[0]
      entry['close_price'] = info[1]
      entry['mark'] = (float(row['bid']) + float(row['ask'])) / 2
      dataset.append(entry)
    except:
      print("exception")

  return dataset

# {'date': '2019-02-16', 'act_symbol': 'EMR', 'expiration': '2019-03-15', 'strike': '67.50', 
# 'call_put': 'Put', 'bid': '0.90', 'ask': '1.00', 'vol': '0.1818', 'delta': '-0.3849', 
# 'gamma': '0.1127', 'theta': '-0.0219', 'vega': '0.0721', 'rho': '-0.0178', 
# 'open_price': 60.22, 'close_price': 59.77, 'mark': 0.95}
