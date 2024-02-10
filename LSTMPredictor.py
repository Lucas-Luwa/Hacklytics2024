import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data from the JSON file
with open('datasetTEST.json', 'r') as f:
    data = json.load(f)

# Prepare the data
StrikeValues = [row['strike'] for row in data]
MarkVals = [row['mark'] for row in data]
histVal = [price for row in data for price in row['history']]
movAvg10 = pd.Series(histVal).rolling(window=10).mean()
delta = [row['delta'] for row in data]
gamma = [row['gamma'] for row in data]
theta = [row['theta'] for row in data]
vega = [row['vega'] for row in data]

features = np.array(list(zip(StrikeValues, MarkVals, movAvg10, delta)))
target = np.array([1 if row['percent_return'] > 0 else 0 for row in data])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
xTrainImputed = imputer.fit_transform(X_train)
xTestImp = imputer.transform(X_test)

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrainImputed)
xTestScaled = scaler.transform(xTestImp)

xTrainReshaped = xTrainScaled.reshape((xTrainScaled.shape[0], 1, xTrainScaled.shape[1]))
xTestReshaped = xTestScaled.reshape((xTestScaled.shape[0], 1, xTestScaled.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(xTrainReshaped.shape[1], xTrainReshaped.shape[2]), return_sequences=True))
model.add(LSTM(40))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(xTrainReshaped, y_train, epochs=25, batch_size=20, validation_split=0.2)

# Evaluate
yPredProbability = model.predict(xTestReshaped)
yPrediction = (yPredProbability > 0.5).astype(int)

# print(y_pred.flatten())
# print(y_test)
equalOrZero = np.logical_or(yPrediction.flatten() == y_test, yPrediction.flatten() == 0)
convToInt = equalOrZero.astype(int)

# Calculate the mean
accuracy1 = np.mean(convToInt)
accuracy2 = np.mean(yPrediction.flatten() == y_test)

print("Accuracy:", accuracy1)
print("OG:", accuracy2)
