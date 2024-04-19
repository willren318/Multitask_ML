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
        label = self.data.iloc[:, -11+i_label]
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
            loss += -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
        loss /= 11
        total_loss += loss
    return total_loss / y_pred.shape[0]


def calculate_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [torch.load(f'models/model{i}.pth') for i in range(11)]

    test_data = pd.read_csv('split_test.csv')
    features = test_data.iloc[:, :-11]
    labels = test_data.iloc[:, -11:].to_numpy()

    result_y_pred = []
    for i in range(11):
        y_preds = []
        for x in features.values:
            x = torch.tensor(x, dtype=torch.float32).reshape((1, -1)).to(device)
            y_pred = models[i](x).flatten()
            # y = 1 if y_pred > 0.5 else 0
            y_preds.append(y_pred.item())

        result_y_pred.append(y_preds)

    result_y_pred = np.array(result_y_pred).T
    return loss_fn(result_y_pred, labels)


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [torch.load(f'models/model{i}.pth') for i in range(11)]

    test_data = pd.read_csv('X_test.csv')

    result_y_pred = []
    for i in range(11):
        y_preds = []
        for x in test_data.values:
            x = torch.tensor(x, dtype=torch.float32).reshape((1, -1)).to(device)
            y_pred = models[i](x).flatten()
            y = 1 if y_pred > 0.5 else 0
            y_preds.append(y)

        result_y_pred.append(y_preds)

    result_y_pred = np.array(result_y_pred).T
    print(result_y_pred)


if __name__ == '__main__':

    calculate_loss()

