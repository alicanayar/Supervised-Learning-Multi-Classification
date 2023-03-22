# pytorch mlp for binary classification
import torch
from numpy import vstack
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear

from torch.nn import Module
from torch.optim import SGD
import torch.nn.functional as F
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss   #search loss function


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path)

 #       df = df.apply(pd.to_numeric)
  #      df = df.dropna()
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset

    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.fc1 = Linear(n_inputs,64)
        self.fc2 = Linear(64, 32)
        self.fc3 = Linear(32, 7)
        self.softmax = torch.nn.Softmax(dim=-1)

    # forward propagate input
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X= self.fc3(X)
        return self.softmax(X)


# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):

        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):

            # clear the gradients
            optimizer.zero_grad()

            # compute the model output
            yhat = model(inputs)
            # calculate loss
            targets = F.one_hot(targets, num_classes=7)
            loss = criterion(yhat, targets)

            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()



# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# prepare the data
if __name__ == '__main__':
    path = "glass.csv"
    train_dl, test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    # define the network
    model = MLP(9)
    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)
    # make a single prediction (expect class=1)
    row = [1.51,12.88,3.43,1.4,73.28,0.69,8.05,0,0.24 ]
    yhat = predict(row, model)
    print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
    row1 = [1,1,1,1,1,1,1,1,1]
    yhat = predict(row1, model)
    print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))


  # ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]
        ####train test birleştir - none değerleri listedden çıkar-neural networkleri 64-64 dene - male female 0 1 dönüştürme döngüsüz