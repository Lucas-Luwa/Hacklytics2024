import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json
from sklearn.model_selection import train_test_split

with open('dataset.json', 'r') as f:
    data = json.load(f)

# X should be (# of points, # of features)
strike_values = [row['strike'] for row in data]
mark_values = [row['mark'] for row in data]
strike_values = np.array(strike_values, dtype=np.float32)
mark_values = np.array(mark_values, dtype=np.float32)
history_values = [row['history'] for row in data]
features = list(zip(strike_values, mark_values))
target = [row['percent_return'] for row in data]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, history_train, history_test = train_test_split(features, target, history_values, test_size=0.2, random_state=42)

X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

train_dataset = TensorDataset(torch.tensor(X_train_np, dtype=torch.float),
                              torch.tensor(y_train_np.reshape((-1, 1)), dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(X_test_np, dtype=torch.float),
                             torch.tensor(y_test_np.reshape((-1, 1)), dtype=torch.float))
train_dataloader = DataLoader(train_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)
max_history_length = 56
history = [history[:max_history_length] + [0] * (max_history_length - len(history)) if len(history) < max_history_length else history[:max_history_length] for history in history_values]
history_train = history[:641]
history_test = history[641:]

class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.lstm1 = nn.LSTM(8, 8)
        self.lstm2 = nn.LSTM(8, 8)
        self.lstm3 = nn.LSTM(8, 8)
        self.lstm4 = nn.LSTM(8, 1)

        self.hidden_layer_1 = nn.Linear(max_history_length + 1, 400)
        self.hidden_1_activation = nn.LeakyReLU()

        self.hidden_layer_2 = nn.Linear(400, 400)
        self.hidden_2_activation = nn.LeakyReLU()

        self.hidden_layer_3 = nn.Linear(400, 400)
        self.hidden_3_activation = nn.LeakyReLU()

        self.out = nn.Linear(400, 1)
        self.out_activation = nn.ReLU()

    def forward(self, x, history):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        combined = torch.cat((x.view(x.size(0), -1), history), dim=1)
        combined = self.hidden_layer_1(combined)
        combined = self.hidden_1_activation(combined)
        combined = self.hidden_layer_2(combined)
        combined = self.hidden_2_activation(combined)
        combined = self.hidden_layer_3(combined)
        combined = self.hidden_3_activation(combined)
        combined = self.out(combined)
        combined = self.out_activation(combined)
        return combined

device = "cpu"
model = MLP1().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        history = torch.tensor(history_train[i * 128: i * 128 + X.shape[0]], dtype=torch.float).to(device)
        y_hat = model(X, history)
        mse = loss_fn(y_hat, y)
        train_loss += mse.item()
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

    num_batches = len(dataloader)
    train_mse = train_loss / num_batches
    print(f'Train RMSE: {train_mse**(1/2)}')

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            history = torch.tensor(history_test[i * 128: i * 128 + X.shape[0]], dtype=torch.float).to(device)
            y_hat = model(X, history)
            test_loss += loss_fn(y_hat, y).item()

    num_batches = len(dataloader)
    test_mse = test_loss / num_batches

    print(f'Test RMSE: {test_mse**(1/2)}\n')


epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}:")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
