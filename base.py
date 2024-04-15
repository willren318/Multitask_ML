import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, r2_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.svm import SVC

x = pd.read_csv('X_train.csv')
y = pd.read_csv('y.csv')

x_train, x_test = train_test_split(x.to_numpy(), test_size=0.2, shuffle=False)
y_train, y_test = train_test_split(y.to_numpy(), test_size=0.2, shuffle=False)


poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)

accuracies = []
f1s = []
recalls = []
precisions = []
models = [AdaBoostClassifier() for _ in range(11)]
for i in range(len(models)):
    models[i] = models[i].fit(x_train_poly, y_train[:, i])
    labels = models[i].predict(x_test_poly)
    accuracies.append(accuracy_score(labels, y_test[:, i]))
    f1s.append(f1_score(labels, y_test[:, i]))
    recalls.append(recall_score(labels, y_test[:, i]))
    precisions.append(precision_score(labels, y_test[:, i]))

print('=' * 20 + 'Accuracies' + '=' * 20)
print(accuracies)
print('=' * 20 + 'F1 scores' + '=' * 20)
print(f1s)
print('=' * 20 + 'recall scores' + '=' * 20)
print(recalls)
print('=' * 20 + 'precision scores' + '=' * 20)
print(precisions)