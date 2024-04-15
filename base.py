import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, r2_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import resample

x = pd.read_csv('X_train.csv')
y = pd.read_csv('y.csv')

accuracies = []
f1s = []
recalls = []
precisions = []
models = [AdaBoostClassifier() for _ in range(11)]

for i in range(11):
    df = pd.concat([x, y.iloc[:, i]], axis=1)
    df_majority = df[df['y_' + str(i)] == 0]
    df_minority = df[df['y_' + str(i)] == 1]

    # 过采样少数类
    df_minority_upsampled = resample(
        df_minority,
        replace=True,  # 样本替换
        n_samples=len(df_majority),  # 匹配多数类的样本数量
        random_state=123)  # 随机数生成器种子

    # 合并多数类和少数类样本得到新的平衡数据集
    df_balanced = pd.concat([df_minority_upsampled, df_majority])
    # 打乱数据集
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    x_train, x_test = train_test_split(df_balanced.iloc[:, :-1].to_numpy(), test_size=0.2, shuffle=False)
    y_train, y_test = train_test_split(df_balanced.iloc[:, -1].to_numpy(), test_size=0.2, shuffle=False)

    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    models[i].fit(x_train_poly, y_train)
    labels = models[i].predict(x_test_poly)
    accuracies.append(accuracy_score(labels, y_test))
    f1s.append(f1_score(labels, y_test))
    recalls.append(recall_score(labels, y_test))
    precisions.append(precision_score(labels, y_test))

print('=' * 20 + 'Accuracies' + '=' * 20)
print(accuracies)
print('=' * 20 + 'F1 scores' + '=' * 20)
print(f1s)
print('=' * 20 + 'recall scores' + '=' * 20)
print(recalls)
print('=' * 20 + 'precision scores' + '=' * 20)
print(precisions)