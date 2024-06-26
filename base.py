import numpy as np
<<<<<<< HEAD
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score, recall_score, precision_score, f1_score, roc_curve, auc
=======
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
>>>>>>> 9dfbdf6 (submit_version)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import resample
import xgboost as xgb
import matplotlib.pyplot as plt

x = pd.read_csv('X_train.csv')
y = pd.read_csv('y_train.csv')
X_test_pred = pd.read_csv('X_test_pred.csv')

# combine x and y for splitting
df = pd.concat([x, y], axis=1)

# split original data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Define classifiers and their parameter grids
classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
    'SVC': (SVC(probability=True, random_state=42), {'C': [0.1, 1, 10], 'kernel': ['rbf']}),
<<<<<<< HEAD
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),


classifier_name = 'RandomForest'  # Example: Switch between 'RandomForest', 'SVC', 'AdaBoost', 'XGBoost'
=======
 
=======
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]})
}

classifier_name = 'RandomForest'  # Switch between 'RandomForest', 'SVC', 'AdaBoost'
>>>>>>> 9dfbdf6 (submit_version)
model, param_grid = classifiers[classifier_name]

accuracies = []
f1s = []
recalls = []
precisions = []
cross_entropy_losses = []

# Define binary cross-entropy loss function
def binary_cross_entropy(y_true, y_prob):
    # Avoid division by zero
    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

best_models = {} # Store the best model for each label
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
    
    # Store the best model
    best_models[column] = grid.best_estimator_
    
    # Prediction using probabilities to find optimal threshold
    probabilities = grid.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[column], probabilities)
    roc_auc = auc(fpr, tpr)

    # Prediction and scoring using optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    labels = (probabilities > optimal_threshold).astype(int)
    
    accuracies.append(accuracy_score(y_test[column], labels))
    f1s.append(f1_score(y_test[column], labels))
    recalls.append(recall_score(y_test[column], labels))
    precisions.append(precision_score(y_test[column], labels))
    
    # Calculate binary cross-entropy loss for test data
    loss = binary_cross_entropy(y_test[column], probabilities)
    cross_entropy_losses.append(loss)

    print("Best parameters for label", column, ":", grid.best_params_)
    print("Optimal threshold for label", column, ":", optimal_threshold)
    print("Binary Cross-Entropy Loss for label", column, ":", loss)
    
    # Store the best model
    best_models[column] = grid.best_estimator_

# Output the performance metrics
print('=' * 40)
print('Accuracies:', accuracies)
print('F1 scores:', f1s)
print('Recall scores:', recalls)
print('Precision scores:', precisions)
print('Binary Cross-Entropy Losses:', cross_entropy_losses)

print(sum(cross_entropy_losses) / len(cross_entropy_losses))

# Predict on the real test data
X_test_pred = X_test_pred[x_train.columns]

print("Training columns:", x_train.columns)
print("Prediction columns:", X_test_pred.columns)
predictions_df = pd.DataFrame(index=X_test_pred.index)
# Iterate over each label and its corresponding best model
for column, model in best_models.items():
    # Get probabilities for the positive class
    probabilities = model.predict_proba(X_test_pred)[:, 1]
    predictions_df[column] = probabilities
    
predictions_df.to_csv('y_pred.csv', index=False)
