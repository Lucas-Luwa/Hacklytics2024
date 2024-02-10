import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("apple_stock_options.csv")
df_np = df.to_numpy()[:, 1:6]

#X should be (# of points, # of features)
X_np = np.array([df_np[:, 0], df_np[:, 1], df_np[:, 2], df_np[:, 4]]).T
y_np = df_np[:, 3]

X_train_np = X_np[:2000]
X_train_np = X_train_np.astype("float64")

y_train_np = y_np[:2000]
y_train_np = y_train_np.astype("float64")

X_test_np = X_np[2000:]
X_test_np = X_test_np.astype("float64")

y_test_np = y_np[2000:]
y_test_np = y_test_np.astype("float64")

train_dataset = TensorDataset(torch.tensor(X_train_np, dtype=torch.float),
                              torch.tensor(y_train_np.reshape((-1, 1)), dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(X_test_np, dtype=torch.float),
                              torch.tensor(y_test_np.reshape((-1, 1)), dtype=torch.float))
train_dataloader = DataLoader(train_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)
class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.hidden_layer_1 = nn.Linear(4, 400)
        self.hidden_1_activation = nn.LeakyReLU()

        self.hidden_layer_2 = nn.Linear(400, 400)
        self.hidden_2_activation = nn.LeakyReLU()

        self.hidden_layer_3 = nn.Linear(400, 400)
        self.hidden_3_activation = nn.LeakyReLU()

        self.out = nn.Linear(400, 1)
        self.out_activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.hidden_1_activation(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_2_activation(x)
        x = self.hidden_layer_3(x)
        x = self.hidden_3_activation(x)
        x = self.out(x)
        x = self.out_activation(x)
        return x

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

        y_hat = model(X)
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
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            test_loss += loss_fn(y_hat, y).item()
    
    num_batches = len(dataloader)
    test_mse = test_loss / num_batches

    print(f'Test RMSE: {test_mse**(1/2)}\n')

epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}:")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

