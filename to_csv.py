import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1 Raw data visualization
# 1.1 Load data
X_train = np.load("X_train.npy")
print(X_train.shape)
y_train = np.load("y_train.npy")
print(y_train.shape)
X_train = pd.DataFrame(X_train, columns=['x_' + str(i) for i in range(X_train.shape[1])], dtype=np.float32)
y_train = pd.DataFrame(y_train, columns=['y_' + str(i) for i in range(y_train.shape[1])], dtype=np.float32)

# 1.2 first impression of these raw data
def visulization_raw_data(df, features_per_plot):
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
        plt.show()
    
visulization_raw_data(X_train,features_per_plot=10)
# visulization_raw_data(y_train,features_per_plot=11)

# 1.3 preprocess raw data given the visualization

'''
# 1.3.1 drop columns:
x_6 as it has 963 'nan' and 37 `0`,
x_12 as it has 16 'nan' and 984 `0`,
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
x_61 as it has 931 `0` and 69 `nan`,
x_66 as it has 932 `0` and 68 `nan`,
x_77 as it has 994 `0` and 6 `nan`,
x_80 as it has 994 `0` and 6 `nan`,
x_87 as it has 998 `nan`
'''

# 1.3.2 fill missing values with mode for binary features and mean(rounded to nearest int) for numerical features

# 1.3.3 how about 1 and 0 are neary half, we can fill missing values with 0??

x.fillna(x.mean(), inplace=True)
x.to_csv('x.csv', index=False)
y.to_csv('y.csv', index=False)