import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import hamming_loss, precision_recall_fscore_support, roc_auc_score
from numpy.random import seed
from tensorflow.random import set_seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed(1)
set_seed(2)

X_train = pd.read_csv('X_train.csv')
y_train = np.load("y_train.npy")
y_train = pd.DataFrame(y_train)
# X_test = pd.read_csv('X_test.csv')
# X_test = X_test[X_train.columns]  # reorder columns in X_test to match X_train


# Function to classify features
def classify_features(df):
    continuous_features = []
    categorical_features = []
    binary_features = []
    for column in df.columns:
        unique_count = df[column].nunique()
        if unique_count == 2:
            binary_features.append(column)
        elif 3 <= unique_count <= 10:
            categorical_features.append(column)
        elif unique_count > 10:
            continuous_features.append(column)
    
    return continuous_features, categorical_features, binary_features

# Function to preprocess features
def preprocess_features(X):
    continuous_features, categorical_features, binary_features = classify_features(X)
    
    # Scale continuous features
    if continuous_features:
        scaler = StandardScaler()
        X[continuous_features] = scaler.fit_transform(X[continuous_features])
    
    # Encode categorical features
    if categorical_features:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_encoded = encoder.fit_transform(X[categorical_features])
        cat_cols = encoder.get_feature_names_out(categorical_features)
        X = X.drop(categorical_features, axis=1)
        X = pd.concat([X, pd.DataFrame(X_encoded, columns=cat_cols)], axis=1)
    
    return X

# Preprocess entire dataset
X_preprocessed = preprocess_features(X_train)

# Split preprocessed data
x_train, x_test, y_train, y_test = train_test_split(X_preprocessed, y_train, test_size=0.2, random_state=42, shuffle=True)

# # resample the minority class
# df_train = pd.concat([x_train, y_train], axis=1)
# df_majority = df_train[df_train[column] == 0]
# df_minority = df_train[df_train[column] == 1]

# # Resample the minority class within the training data
# df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
# df_balanced = pd.concat([df_minority_upsampled, df_majority]).sample(frac=1, random_state=42)

# # Split the balanced data into features and target variable again
# x_train_balanced = df_balanced.drop(column, axis=1)
# y_train_balanced = df_balanced[column]




# # Function to calculate the class weights
# def compute_class_weight(pos_cases):
#     total_cases = 1000  # total number of cases per label
#     neg_cases = total_cases - pos_cases
#     weight_for_0 = (1 / neg_cases) * (total_cases) / 2.0 
#     weight_for_1 = (1 / pos_cases) * (total_cases) / 2.0
#     return weight_for_0, weight_for_1

# # Custom weighted binary crossentropy loss
# def weighted_binary_crossentropy(y_true, y_pred):
#     class_weight_0_tensor = tf.cast(class_weight_0, dtype=tf.float32)
#     class_weight_1_tensor = tf.cast(class_weight_1, dtype=tf.float32)
    
#     # Ensure that y_true is a float32 tensor for consistent type operations
#     y_true_float = tf.cast(y_true, dtype=tf.float32)

#     # Calculate the weights for each class
#     weights = y_true_float * class_weight_1_tensor + (1. - y_true_float) * class_weight_0_tensor
#     # Calculate the binary crossentropy
#     bce = K.binary_crossentropy(y_true_float, y_pred)
#     # Apply the weights
#     weighted_bce = weights * bce
#     return K.mean(weighted_bce)

# # Assuming an average of 25% positive cases per label
# class_weight_0, class_weight_1 = compute_class_weight(250)



# def calculate_sample_weights(y_train, positive_weight=4.0, negative_weight=1.0):
#     # This function calculates sample weights based on label imbalances.
#     # `y_train` is expected to be a DataFrame or a 2D numpy array with one column per label.
#     weights = np.ones(y_train.shape)
    
#     # Assign weights
#     for label in range(y_train.shape[1]):
#         pos_indices = y_train[:, label] == 1
#         neg_indices = y_train[:, label] == 0
#         weights[pos_indices, label] = positive_weight
#         weights[neg_indices, label] = negative_weight
    
#     # Return the average weight across all labels for each sample
#     return np.mean(weights, axis=1)

# y_train_np = y_train.to_numpy()
# # Calculate sample weights
# sample_weights = calculate_sample_weights(y_train_np)

def custom_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    # Calculate binary crossentropy for each label
    binary_crossentropy_per_label = -y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) - (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    
    # Average across all labels (axis=-1 ensures it averages across the label dimension)
    average_per_sample = tf.reduce_mean(binary_crossentropy_per_label, axis=-1)
    
    # Explicitly average across all samples in the batch (normally handled by Keras, but included here for explicit formula adherence)
    return tf.reduce_mean(average_per_sample)  # This averages over the batch dimension


# Build and train model
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='sigmoid')  # sigmoid activation for binary classification
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=custom_binary_crossentropy, metrics=[
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ])
    
    return model

model = build_model(input_dim=x_train.shape[1], output_dim=y_train.shape[1])
# history = model.fit(x_train, y_train, epochs=300, batch_size=32, sample_weight=sample_weights, validation_data=(x_test, y_test))
history = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test))

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Making predictions and evaluating the model
y_test_pred = model.predict(x_test)
y_test_pred_binary = (y_test_pred > 0.3).astype(int)

h_loss = hamming_loss(y_test, y_test_pred_binary)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_test_pred_binary, average='samples')

print(f"Hamming Loss: {h_loss}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")

# Save predictions
pd.DataFrame(y_test_pred).to_csv('predictions_probabilities.csv', index=False)
pd.DataFrame(y_test_pred_binary).to_csv('predictions_binary.csv', index=True)

print(y_test)
print(pd.DataFrame(y_test_pred_binary))