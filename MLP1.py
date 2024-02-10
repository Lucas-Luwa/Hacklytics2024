import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_json("dataset.json")
df_np = df.to_numpy()
strikes = df_np[:, 3]
call_put = np.array(pd.factorize(df_np[:, 4])[0].tolist())
bid = df_np[:, 5]
ask = df_np[:, 6]
vol = df_np[:, 7]
delta = df_np[:, 8]
gamma = df_np[:, 9]
theta = df_np[:, 10]
vega = df_np[:, 11]
rho = df_np[:, 12]
open_price = df_np[:, 13]
history = df_np[:, 14]
percent_return = df_np[:, 15]
# #X should be (# of points, # of features)
X_np = np.array([strikes, call_put, bid, ask, vol, delta, gamma, theta, vega, rho, open_price]).T
# X_np = np.array([vega, rho, open_price]).T
y_np = rho

X_train_np = X_np[:700]
X_train_np = X_train_np.astype("float64")

y_train_np = y_np[:700]
y_train_np = y_train_np.astype("float64")

X_test_np = X_np[700:]
X_test_np = X_test_np.astype("float64")

y_test_np = y_np[700:]
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

		# initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=11, out_features=500)
        self.sigmoid1 = nn.Sigmoid()
		# initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid1(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.softmax(x)
        # return the output predictions
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP1().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(dataloader, model, loss_fn, optimizer):
    model.train()

    totalTrainLoss = 0
    
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    
    # loop over the training set
    for (x, y) in train_dataloader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = loss_fn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
    return totalTrainLoss, trainCorrect

def test(dataloader, model, loss_fn):
    totalValLoss = 0
    valCorrect = 0
    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in test_dataloader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += loss_fn(pred, y)
            # calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
    return totalValLoss, valCorrect

epochs = 10
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}
for epoch in range(epochs):
    totalTrainLoss, trainCorrect = train(train_dataloader, model, loss_fn, optimizer)
    totalValLoss, valCorrect = test(test_dataloader, model, loss_fn)
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(train_dataloader.dataset)
    valCorrect = valCorrect / len(test_dataloader.dataset)
    # update our training history
    # H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    # H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(epoch + 1, epochs))
    print("Train accuracy: {:.4f}".format(trainCorrect))
    print("Val accuracy: {:.4f}\n".format( valCorrect))