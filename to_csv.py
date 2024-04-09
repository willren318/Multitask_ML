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
                df[column].replace('', np.nan, inplace=True)
                sns.histplot(df[column].dropna(), bins=30, kde=False)
                plt.title(column)

        plt.tight_layout()
        plt.show()
    
visulization_raw_data(X_train,features_per_plot=10)
# visulization_raw_data(y_train,features_per_plot=11)

# 1.3 preprocess raw data given the visualization
# 1.3.1 drop columns with too many missing values



x.fillna(x.mean(), inplace=True)
x.to_csv('x.csv', index=False)
y.to_csv('y.csv', index=False)