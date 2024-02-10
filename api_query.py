import requests
import json
from startEndPrices import getOpenClosePrice
from getDataset import options_history
from optPercentReturn import calcPercentReturn

def get_full_dataset(output_file):
    endpoint = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
    query = "SELECT * FROM `option_chain` WHERE `date` >= '2023-01-01' AND `expiration` < '2024-02-10' ORDER BY `date` ASC"
    res = requests.get(endpoint, params={'q': query})
    data = res.json()['rows']

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
        except Exception as e:
            print("exception:", e)

    with open(output_file, 'w') as f:
        json.dump(dataset, f)

    print("Dataset saved to", output_file)

output_file = 'dataset.json'
get_full_dataset(output_file)

# {'date': '2023-01-02', 'act_symbol': 'ABG', 'expiration': '2023-01-20', 'strike': '210.00', 
#  'call_put': 'Put', 'bid': '28.70', 'ask': '32.70', 'vol': '0.3293', 'delta': '-0.9972', 
#  'gamma': '0.0033', 'theta': '0.0000', 'vega': '0.0052', 'rho': '0.0000', 
#  'open_price': 180.06, 'close_price': 184.04, 'mark': 30.700000000000003, 
#  'history': [180.05999755859375, 178.44000244140625, 177.50999450683594, ..., 183.11749267578125, 183.38999938964844], 
#  'percent_return': -15.44}
