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
from imblearn.over_sampling import SMOTE, RandomOverSampler


# Set random seed for reproducibility
seed(1)
set_seed(2)

X_train = pd.read_csv('X_train.csv')
y_train = np.load("y_train.npy")
y_train = pd.DataFrame(y_train)
# X_test = pd.read_csv('X_test.csv')
# X_test = X_test[X_train.columns]  # reorder columns in X_test to match X_train


# Split preprocessed data
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

def apply_random_oversampling(X, y):
    ros = RandomOverSampler(random_state=42)
    y_resampled = pd.DataFrame(index=X.index)

    # Initialize a flag to check if X has been resampled
    X_resampled = None

    # Apply Random Over-Sampling to each label independently
    for column in y.columns:
        y_column = y[column]
        # Ensure y_column is in the right format
        if y_column.ndim == 1 or (y_column.ndim == 2 and y_column.shape[1] == 1):
            if X_resampled is None:  # Only resample X once using the first label
                X_res, y_res = ros.fit_resample(X, y_column)
                X_resampled = X_res
            else:
                _, y_res = ros.fit_resample(X, y_column)
            y_resampled[column] = y_res
        else:
            raise ValueError(f"Incorrect format for label {column}")

    return X_resampled, y_resampled

# Example usage:
x_train_resampled, y_train_resampled = apply_random_oversampling(x_train, y_train)


# Number of samples added
num_samples_added = x_train_resampled.shape[0] - x_train.shape[0]
print(f"Number of samples added: {num_samples_added}")

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
    
    return X, continuous_features, binary_features



# Preprocess entire dataset
x_train, continuous_features, binary_features = preprocess_features(x_train_resampled)

# Preprocess entire dataset
X_test, continuous_features, binary_features = preprocess_features(x_test)

y_train = y_train_resampled


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
        Dense(32, activation='relu'),
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
