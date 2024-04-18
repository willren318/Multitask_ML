import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import resample
import xgboost as xgb
import matplotlib.pyplot as plt

x = pd.read_csv('X_train.csv')
y = pd.read_csv('y.csv')

# Combine x and y for splitting
df = pd.concat([x, y], axis=1)

# Split original data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Define classifiers and their parameter grids
classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
    'SVC': (SVC(probability=True, random_state=42), {'C': [0.1, 1, 10], 'kernel': ['rbf']}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'GradBoost': (GradientBoostingClassifier(), {'learning_rate': [0.1, 0.01], 'n_estimators': [100, 200]}),
    'XGBoost':(xgb.XGBClassifier(), {'objective': ['multi:softmax'], 'num_class':[2], 'eval_metric':["mlogloss"]})

}

classifier_name = 'XGBoost'  # Example: Switch between 'RandomForest', 'SVC', 'AdaBoost'
model, param_grid = classifiers[classifier_name]

accuracies = []
f1s = []
recalls = []
precisions = []

for column in y_train.columns:
    # Create a DataFrame for each label within the training data
    df_train = pd.concat([x_train, y_train[column]], axis=1)
    df_majority = df_train[df_train[column] == 0]
    df_minority = df_train[df_train[column] == 1]

    # Resample the minority class within the training data
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
    df_balanced = pd.concat([df_minority_upsampled, df_majority]).sample(frac=1, random_state=42)

    # Split the balanced data into features and target variable again
    x_train_balanced = df_balanced.drop(column, axis=1)
    y_train_balanced = df_balanced[column]
    

    # grid search
    grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', verbose=1)
    grid.fit(x_train_balanced, y_train_balanced)
    
    # Prediction using probabilities to find optimal threshold
    probabilities = grid.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[column], probabilities)
    roc_auc = auc(fpr, tpr)

    # Prediction and scoring
    
    # labels = grid.predict(x_test)
    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    labels = (probabilities > optimal_threshold).astype(int)
    
    accuracies.append(accuracy_score(y_test[column], labels))
    f1s.append(f1_score(y_test[column], labels))
    recalls.append(recall_score(y_test[column], labels))
    precisions.append(precision_score(y_test[column], labels))

    print("Best parameters for label", column, ":", grid.best_params_)
    print("Optimal threshold for label", column, ":", optimal_threshold)

# Output the performance metrics
print('=' * 40)
print('Accuracies:', accuracies)
print('F1 scores:', f1s)
print('Recall scores:', recalls)
print('Precision scores:', precisions)




# y_true = []
# y_pred = []

# # Inference
# def loss_fn(y_pred, y_true):
#     total_loss = 0
#     for i in range(y_pred.shape[0]):
#         loss = 0
#         for j in range(y_true.shape[1]):
#             y, y_hat = y_true[i, j], y_pred[i, j]
#             loss += -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
#         loss /= 11
#         total_loss += loss
#     return total_loss / y_pred.shape[0]

# labels = []
# for i in range(11):
#     label = models[i].predict(x)
#     labels.append(label)
# labels = np.array(labels)

# print('=' * 20 + 'Total Loss' + '=' * 20)
# print(loss_fn(labels, y_inference))


