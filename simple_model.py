import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# {'date': '2023-01-02', 'act_symbol': 'ABG', 'expiration': '2023-01-20', 'strike': '210.00', 
#  'call_put': 'Put', 'bid': '28.70', 'ask': '32.70', 'vol': '0.3293', 'delta': '-0.9972', 
#  'gamma': '0.0033', 'theta': '0.0000', 'vega': '0.0052', 'rho': '0.0000', 
#  'open_price': 180.06, 'close_price': 184.04, 'mark': 30.700000000000003, 
#  'history': [180.05999755859375, 178.44000244140625, 177.50999450683594, ..., 183.11749267578125, 183.38999938964844], 
#  'percent_return': -15.44}

# code based on https://medium.com/@fhuqtheta/applications-of-machine-learning-in-options-trading-b416c5a67831

# Load options trading data
with open('dataset.json', 'r') as f:
    data = json.load(f)
# Prepare the data
strike_values = [row['strike'] for row in data]
mark_values = [row['mark'] for row in data]
history_values = [price for row in data for price in row['history']]
moving_average_10 = pd.Series(history_values).rolling(window=10).mean()
features = list(zip(strike_values, mark_values, moving_average_10))
target = [1 if row['percent_return'] > 0 else 0 for row in data]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Create and train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Make predictions on the test set
predictions = clf.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
