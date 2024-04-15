import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, r2_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import resample

x = pd.read_csv('X_train.csv')
y = pd.read_csv('y.csv')

x, x_inference = x.iloc[:800, :], x.iloc[800:, :].to_numpy()
y, y_inference = y.iloc[:800, :], y.iloc[800:, :].to_numpy()

accuracies = []
f1s = []
recalls = []
precisions = []
models = [RandomForestClassifier()] * 11

y_true = []
y_pred = []


for i in range(11):
    df = pd.concat([x, y.iloc[:, i]], axis=1)
    df_majority = df[df['y_'+str(i)] == 0]
    df_minority = df[df['y_'+str(i)] == 1]

    # 过采样少数类
    df_minority_upsampled = resample(
        df_minority,
         replace=True,     # 样本替换
         n_samples=len(df_majority),  # 匹配多数类的样本数量
         random_state=123)  # 随机数生成器种子

    # 合并多数类和少数类样本得到新的平衡数据集
    df_balanced = pd.concat([df_minority_upsampled, df_majority])
    # 打乱数据集
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    x_train, x_test = train_test_split(df_balanced.iloc[:, :-1].to_numpy(), test_size=0.2, shuffle=False)
    y_train, y_test = train_test_split(df_balanced.iloc[:, -1].to_numpy(), test_size=0.2, shuffle=False)

    models[i].fit(x_train, y_train)
    labels = models[i].predict(x_test)


    accuracies.append(accuracy_score(labels, y_test))
    f1s.append(f1_score(labels, y_test))
    recalls.append(recall_score(labels, y_test))
    precisions.append(precision_score(labels, y_test))

    joblib.dump(models[i], 'models/random_forest_model.joblib')


print('=' * 20 + 'Accuracies' + '=' * 20)
print(accuracies)
print('=' * 20 + 'F1 scores' + '=' * 20)
print(f1s)
print('=' * 20 + 'recall scores' + '=' * 20)
print(recalls)
print('=' * 20 + 'precision scores' + '=' * 20)
print(precisions)

# Inference
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

labels = []
for i in range(11):
    label = models[i].predict(x)
    labels.append(label)
labels = np.array(labels)

print('=' * 20 + 'Total Loss' + '=' * 20)
print(loss_fn(labels, y_inference))

