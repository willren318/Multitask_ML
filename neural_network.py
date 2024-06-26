import json

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, r2_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import resample
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data_file, i_label):
        self.data = pd.read_csv(data_file)
        features = self.data.iloc[:, 0:-11]
        label = self.data.iloc[:, -11 + i_label]
        self.data = pd.concat([features, label], axis=1)
        if 'train' in data_file:
            i_label = 'y_' + str(i_label)
            df_majority = self.data[self.data[i_label] == 0]
            df_minority = self.data[self.data[i_label] == 1]

            # Resample the minority class within the training data
            df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
            df_balanced = pd.concat([df_minority_upsampled, df_majority]).sample(frac=1, random_state=42)
            self.data = df_balanced
        self.data = self.data.to_numpy()

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features = torch.from_numpy(self.data[idx][:-1]).float()
        labels = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        return features, labels

class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def loss_fn(y_pred, y_true):
    total_loss = 0
    for i in range(y_pred.shape[0]):
        loss = 0
        for j in range(y_true.shape[1]):
            y, y_hat = y_true[i, j], y_pred[i, j]
            loss += -y * torch.log(y_hat) - (1 - y) * torch.log(1 - y_hat)
        loss /= 11
        total_loss += loss
    return total_loss / y_pred.shape[0]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [LinearModel(input_size=34).to(device)] * 11

accuracies = []
f1s = []
precisions = []
recalls = []

for i in range(1):
    epochs = 100
    batch_size = 32
    train_set = MyDataset('split_train.csv', i_label=i)
    test_set = MyDataset('split_test.csv', i_label=i)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    models[i].train()
    loss_FN = nn.MSELoss()
    optimizer = optim.SGD(models[i].parameters(), lr=0.001, momentum=0.9)

    bar = tqdm(range(epochs))
    for epoch in bar:
        losses = []
        for x, y_true in train_loader:
            x, y = x.to(device), y_true.to(device)
            y_pred = models[i](x).flatten()
            optimizer.zero_grad()
            # loss = loss_fn(y_pred, y_true)
            loss = loss_FN(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(y_pred, y[:, 0])
            # print(loss.item())
        # print(sum(losses) / len(losses))
        bar.set_description(f'Mean loss: {sum(losses) / len(losses)}')

    torch.save(models[i], f'models/model{i}.pth')

    models[i].eval()
    y_preds = []
    y_trues = []
    acc_sum = 0
    for x, y_true in test_loader:
        x, y_true = x.to(device), y_true.to(device)
        y_pred = models[i](x).flatten()
        # y_pred = 1 if y_pred > 0.5 else 0
        y = 1 if y_pred > 0.5 else 0
        y_preds.append(y)
        y_trues.append(y_true.item())
        print(y_pred, y, y_true.item())


    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)

    accuracy = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)

    accuracies.append(accuracy)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    print(f'accuracy = {accuracy}')
    print(f'f1_score = {f1}')
    print(f'precision_score = {precision}')
    print(f'recall_score = {recall}')

json.dump(accuracies, open('results/accuracies.json', 'w'))
json.dump(f1s, open('results/f1s.json', 'w'))
json.dump(precisions, open('results/precisions.json', 'w'))
json.dump(recalls, open('results/recalls.json', 'w'))




