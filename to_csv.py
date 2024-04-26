import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# 1 Raw data visualization
# 1.1 Load data
X_train = np.load("X_train.npy")
print(X_train.shape)

y_train = np.load("y_train.npy")
print(y_train.shape)
X_test_pred = np.load("X_test.npy")

X_train = pd.DataFrame(X_train, columns=['x_' + str(i) for i in range(X_train.shape[1])], dtype=np.float32)
y_train = pd.DataFrame(y_train, columns=['y_' + str(i) for i in range(y_train.shape[1])], dtype=np.float32)
X_test_pred = pd.DataFrame(X_test_pred, columns=['x_' + str(i) for i in range(X_test_pred.shape[1])], dtype=np.float32)

# create a directory to save plots if it doesn't exist
directory = 'data_plots'
if not os.path.exists(directory):
    os.makedirs(directory)
    
# 1.2 first impression of these raw data
def visulization_raw_data(df, features_per_plot, dataset_name):
    # Create a directory to plot files if it doesn't exist

    total_features = df.columns
    num_plots = np.ceil(len(total_features) / features_per_plot).astype(int)

    for plot_index in range(num_plots):
        plt.figure(figsize=(20, 10))
        start_index = plot_index * features_per_plot
        end_index = start_index + features_per_plot
        features_subset = total_features[start_index:end_index]

        for i, column in enumerate(features_subset, 1):
            plt.subplot(2, 5, i)  # Layout for 10 subplots (2 rows x 5 columns)
            if df[column].nunique() == 2:
                # for binary features, use a bar chart
                df[column].fillna('Blank').value_counts().plot(kind='bar')
                plt.title(column)
            else:
                # for numerical features, use a histogram
                df[column].replace('', np.nan, inplace=True)  # Replace blanks with NaN
                cleaned_data = df[column].dropna()  # Drop NaN values for histogram
                sns.histplot(cleaned_data, bins=30, kde=False)
                plt.title(column)

                # Count the number of blank or NaN values and display it on the plot
                blank_or_nan_count = df[column].isna().sum()
                plt.annotate(f'Blanks/Nan: {blank_or_nan_count}', xy=(0.7, 0.9), xycoords='axes fraction',
                            fontsize=12, bbox=dict(boxstyle="round", fc="yellow"))

        plt.tight_layout()
        file_path = os.path.join(directory, f'{dataset_name}_{plot_index + 1}.png')
        # Save each plot to directory
        plt.savefig(file_path)
        plt.show()
    
visulization_raw_data(X_train, features_per_plot=10, dataset_name='X_train')
visulization_raw_data(y_train,features_per_plot=10, dataset_name='y_train')

# 2 preprocess raw data given the visualization

# 2.1 drop columns with too many missing values:
# define a function to determine if a column should be dropped based on the count of NaN and 0 values
def should_drop(column):
    nan_count = column.isna().sum()
    zero_count = (column == 0).sum()
    if nan_count + zero_count >= 998:
        return True
    return False

# identify columns to drop and drop them
columns_to_drop = [col for col in X_train.columns if should_drop(X_train[col])]
X_train.drop(columns=columns_to_drop, inplace=True)
X_test_pred.drop(columns=columns_to_drop, inplace=True)

'''
columns_to_drop:
x_6 as it has 963 'nan' and 37 `0`,
x_12 as it has 984 `0` and 16 'nan',
x_16 as it has 984 `0` and 16 'nan',
x_17 as it has 983 `0` and 16 'nan' and 1 `1`,
x_18 as it has 988 `0` and 12 'nan',
x_19 as it has 987 `0` and 12 'nan' and 1 `1`,
x_21 as it has 988 `0` and 12 'nan',
x_22 as it has 986 `0` and 12 'nan' and 2 `1`,
x_23 as it has 988 `0` and 12 'nan',
x_24 as it has 987 `0` and 12 'nan' and 1 `1`,
x_31 as it has 997 `0` and 3 'nan',
x_40 as it has 991 `0` and 7 'nan' and 2 `1`,
x_47 as it has 999 `0` and only 1 `1`,
x_51 as it has 904 `0` and 96 `nan`,
x_55 as it has 930 `0` and 69 'nan' and 1 `1`,
x_60 as it has 930 `0` and 69 'nan' and 1 `1`,
x_61 as it has 931 `0` and 69 `nan`,
x_62 as it has 930 `0` and 69 'nan' and 1 `1`,
x_63 as it has 931 `0` and 68 'nan' and 1 `1`,
x_65 as it has 930 `0` and 68 'nan' and 2 `1`,
x_66 as it has 932 `0` and 68 `nan`,
x_69 as it has 930 `0` and 68 'nan' and 2 `1`,
x_77 as it has 994 `0` and 6 `nan`,
x_80 as it has 994 `0` and 6 `nan`,
x_87 as it has 998 `nan`
'''

# 2.2 try to fill missing values with different methods

# 2.2.1 fill with mode for binary features and mean(rounded to nearest int) for numerical features
def fill_missing_values_mode_mean(df):
    filled_df = df.copy()
    for column in filled_df.columns:
        if filled_df[column].dropna().isin([0, 1]).all():
            mode = filled_df[column].mode().iloc[0]
            filled_df[column].fillna(mode, inplace=True)
        else:
            mean_value = int(round(filled_df[column].mean()))
            filled_df[column].fillna(mean_value, inplace=True)
    return filled_df

# 2.2.2 fill missing value using KNN imputation
def fill_missing_values_knn(df, n_neighbors=5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Initialize and apply the KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_scaled = imputer.fit_transform(df_scaled)
    

    # Inverse transform to original scale and convert back to DataFrame
    df_imputed = scaler.inverse_transform(df_imputed_scaled)
    df_imputed = np.rint(df_imputed)
    df_imputed_df = pd.DataFrame(df_imputed, columns=df.columns)

    return df_imputed_df

# switch between different methods via uncommenting one of the following lines:
# X_train = fill_missing_values_mode_mean(X_train)
X_train = fill_missing_values_knn(X_train)
X_test_pred = fill_missing_values_knn(X_test_pred)

# visualize the data after filling missing values
visulization_raw_data(X_train,features_per_plot=10, dataset_name='X_train_filled')


# 3 correlation analysis between features and target
# compute and visualize the correlation matrix
def visualize_correlations(df, dataset_name):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    
    file_path = os.path.join(directory, f'{dataset_name}_correlation.png')
    # Save each plot to directory
    plt.savefig(file_path)
    
    plt.show()

# visualize the correlation matrix for X_train and y_train
visualize_correlations(X_train, dataset_name='X_train')
visualize_correlations(y_train, dataset_name='y_train')

# 4 feature importance using random forest
def feature_importance(X_train, y_train):
    model = RandomForestClassifier()  # Use RandomForestRegressor for a regression problem
    model.fit(X_train, y_train)

    # Get feature importance
    importances = model.feature_importances_
    features = X_train.columns
    feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 16))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    
    file_path = os.path.join(directory, f'features_importance.png')
    # Save each plot to directory
    plt.savefig(file_path)
    
    plt.show()
    
    return feature_importance, feature_importance[feature_importance < 0.01].index.tolist()
    
features_importance, low_importance_features = feature_importance(X_train, y_train)

# drop features with importance less than 0.01
X_train = X_train[features_importance[features_importance >= 0.01].index]

# 5 save the preprocessed data to csv files 'X_train.csv' and 'y_train.csv'
X_train.to_csv('X_train.csv', index=False)

y_train.to_csv('y_train.csv', index=False)


# final columns to drop
final_columns_to_drop = columns_to_drop + low_importance_features

visualize_correlations(X_train, dataset_name='X_train')

X_test_pred.drop(columns=low_importance_features, inplace=True)
X_test_pred.to_csv('X_test_pred.csv', index=False)