# CodeRunner Multi-Class Classification

## Cleaning data: to_csv.py

Load data from numpy binary files of 'X_train.npy', 'y_train.npy' and 'X_test.npy' and create pandas Dataframes for them. 

#### Drop columns with too many missing values

1. Count the number of blank or zero values for every feature and show specific values on plots. 
2. Drop columns with too many missing values according to these plots.  

#### Fill Missing Values

1. fill with mode for binary features and mean(rounded to nearest int) for numerical features
2. fill missing value using KNN imputation


#### Correlation analysis between features and target

1. Get feature importance using random forest and visualize the results.
2. drop features with importance less than 0.01.

#### Save the cleaned data  

After dropping, there are 34 features kept, we save them to csv files 'X_train.csv', and save labels to 'y_train.csv'

## Model selection: base.py

Use Grid View Cross Validation to select the best model and its hyperparameter in following models: RandomForest, SVC, AdaBoost, GradBoost, XGBoost  

1. Split dataset into training set(80%) and testing set(20%).

2. Because there are 11 class to be classified, we create 11 classifier to judge them individually.

3. For every class, the proportion of positive and negative samples is inconsistent. We found that positive samples less than negative, so we resample positive sample by random sampling so that the quantity and the negative sample are kept constant.

4. Train 11 models on training set by Grad View Cross-Validation, obtain the best hyperparameters for every model and calculate their Binary Cross-Entropy Loss, Accuracy, F1 Score, Precision Score, and Recall.

## Neural Network: neural_network_model.py

Train a Multi Layer Perceptron model to solve this multi task problem. 

1. Define a Model with three fully connected layers and activate output of the last layer by sigmoid activation function to make the predict value between 0 and 1.
2. Define our total binary cross entropy loss function to update model parameters
3. Select better optimizer in SGD, RMSprop and Adam.
4. Train model for 300 epochs and evaluate the performance. 


## Prediction file: y_pred.csv
Prediction for test data
