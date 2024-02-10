import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import json
import numpy as np

with open('datasetTEST.json', 'r') as f:
    data = json.load(f)

def model(window):
    # Prepare the data
    strike_values = [row['strike'] for row in data]
    mark_values = [row['mark'] for row in data]
    history_values = [price for row in data for price in row['history']]
    moving_average_10 = pd.Series(history_values).rolling(window).mean()
    delta = [row['delta'] for row in data]
    gamma = [row['gamma'] for row in data]
    theta = [row['theta'] for row in data]
    vega = [row['vega'] for row in data]

    # print(len(vega), len(theta), len(gamma), len(delta), len(moving_average_10), len(mark_values), len(strike_values))

    features = list(zip(strike_values, mark_values, moving_average_10, delta))
    target = [1 if row['percent_return'] > 0 else 0 for row in data]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Create and train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_imputed, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test_imputed)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy:", accuracy)
    return accuracy

vals = np.zeros((100,))
maxVal = -1
index = 0
for i in range(100):
    vals[i] = model(i)
    if vals[i] > maxVal:
        maxVal = vals[i]
        index = i

print(maxVal)

print(vals)
print(np.mean(vals))
print(np.max(vals))
print(index)
#6-7 Have the best results 