import sklearn
import sklearn.ensemble
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

x = pd.read_csv('x.csv')
y = pd.read_csv('y.csv')
x.fillna(0, inplace=True)

x_train, x_test = train_test_split(x.to_numpy(), test_size=0.2, shuffle=False)
y_train, y_test = train_test_split(y.to_numpy(), test_size=0.2, shuffle=False)

poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)

accuracies = []
r2s = []
models = [sklearn.ensemble.AdaBoostClassifier() for _ in range(11)]
for i in range(len(models)):
    models[i] = models[i].fit(x_train_poly, y_train[:, i])
    labels = models[i].predict(x_test_poly)
    accuracies.append(accuracy_score(labels, y_test[:, i]))
    r2s.append(r2_score(labels, y_test[:, i]))

print('=' * 20 + 'accuracies' + '=' * 20)
print(accuracies)
print('=' * 20 + 'R2 scores' + '=' * 20)
print(r2s)