# CodeRunner Multi-Class Classification

## to_csv.py

1 Raw data visualization
    1.1 Load data
    1.2 first impression of these raw data

2 preprocess raw data given the visualization
    2.1 drop columns with too many missing values
    2.2 try to fill missing values with different methods
        2.2.1 fill with mode for binary features and mean(rounded to nearest int) for numerical features
        2.2.2 fill missing value using KNN imputation

3 correlation analysis between features and target

4 feature importance using random forest and drop features with importance less than 0.01

5 save the preprocessed data to csv files 'X_train.csv' and 'y_train.csv' 

## base.py

划分输入数据，做二项式扩展，创建11个相同的分类器AdsBoost，训练出11个分类器，分别计算准确率和r2 score.