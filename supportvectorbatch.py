# -*- coding: utf-8 -*-
"""supportVectorbatch.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S1_emCinPBFQiDo3_YZHFto1jocnRrbR
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skimage.io import imread
from skimage.transform import resize
import os

# Directories and file paths
train_dir = 'train_set\\images'
val_dir = 'val_set\\images'
test_dir = 'test_set\\images'
train_ann_file = 'train_annotation.csv'
val_ann_file = 'val_annotation.csv'
test_ann_file = 'test_annotation.csv'
img_width = 227
img_height = 227
batch_size = 20
num_emotions = 8

# One-hot encode emotion labels
def one_hot_encode(number, num_classes=num_emotions):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[number] = 1
    return one_hot_vector

# Function to load and preprocess a batch of data
def load_batch(dir_path, ann_file, batch_paths, dims):
    images = []
    valences = []
    arousals = []
    emotions = []

    anno = pd.read_csv(ann_file)

    for filename in batch_paths:
        img_path = os.path.join(dir_path, filename)
        img = imread(img_path)
        img_resized = resize(img, (img_height, img_width))
        images.append(img_resized.flatten())  # Flatten image to 1D array

        row = anno[anno["filename"] == int(filename.split(".")[0])]
        if not row.empty:
            valences.append(row["Valance"].values[0])
            arousals.append(row["Arousal"].values[0])
            emotions.append(one_hot_encode(row["Expression"].values[0]))

    images = np.asarray(images)
    valences = np.asarray(valences)
    arousals = np.asarray(arousals)
    # print(emotions)
    emotions = np.asarray(emotions)

    return images, valences, arousals, emotions

# concordance_correlation_coefficient
def CCC(y_true, y_pred):
    """Calculate the Concordance Correlation Coefficient (CCC) between true and predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2)

    return ccc

# Function to generate batches of data
def data_generator(dir_path, ann_file, images, batch_size):
    while True:
        batch_paths = np.random.choice(images, size=batch_size)
        yield load_batch(dir_path, ann_file, batch_paths, (img_height, img_width))

# Get the list of image filenames
train_img = [f for f in os.listdir(train_dir)]
val_img = [f for f in os.listdir(val_dir)]
test_img = [f for f in os.listdir(test_dir)]

# Initialize generators
train_gen = data_generator(train_dir, train_ann_file, train_img, batch_size)
val_gen = data_generator(val_dir, val_ann_file, val_img, batch_size)
test_gen = data_generator(test_dir, val_ann_file, test_img, batch_size)

# Standardize features
scaler = StandardScaler()

# Create SVM models
svr_valence = SVR(kernel='rbf')
svr_arousal = SVR(kernel='rbf')
svc_emotion = SVC(kernel='rbf', probability=True)

# Batch training
num_batches = len(train_img) // batch_size

for _ in range(3):
    X_train_batch, y_train_valence_batch, y_train_arousal_batch, y_train_emotion_batch = next(train_gen)

    X_train_scaled_batch = scaler.fit_transform(X_train_batch)

    # print(y_train_arousal_batch)

    svr_valence.fit(X_train_scaled_batch, y_train_valence_batch)
    svr_arousal.fit(X_train_scaled_batch, y_train_arousal_batch)
    svc_emotion.fit(X_train_scaled_batch, np.argmax(y_train_emotion_batch, axis=1))

# Evaluate on validation set
X_val, y_val_valence, y_val_arousal, y_val_emotion = next(val_gen)
X_val_scaled = scaler.transform(X_val)

valence_predictions = svr_valence.predict(X_val_scaled)
arousal_predictions = svr_arousal.predict(X_val_scaled)
emotion_predictions = svc_emotion.predict(X_val_scaled)

# Calculate evaluation metrics
valence_mse = np.mean((valence_predictions - y_val_valence) ** 2)
arousal_mse = np.mean((arousal_predictions - y_val_arousal) ** 2)
emotion_accuracy = np.mean(emotion_predictions == y_val_emotion.argmax(axis=1))

print(f'Valence MSE: {valence_mse:.4f}')
print(f'Arousal MSE: {arousal_mse:.4f}')
print(f'Emotion Accuracy: {emotion_accuracy:.4f}')

# Evaluate on test set
X_test, y_test_valence, y_test_arousal, y_test_emotion = next(test_gen)
X_test_scaled = scaler.transform(X_test)

test_valence_predictions = svr_valence.predict(X_test_scaled)
test_arousal_predictions = svr_arousal.predict(X_test_scaled)
test_emotion_predictions = svc_emotion.predict(X_test_scaled)

# Calculate test set evaluation metrics
test_valence_mse = np.mean((test_valence_predictions - y_test_valence) ** 2)
test_arousal_mse = np.mean((test_arousal_predictions - y_test_arousal) ** 2)
test_emotion_accuracy = np.mean(test_emotion_predictions == y_test_emotion.argmax(axis=1))
test_valenece_ccc = CCC(y_test_valence, test_valence_predictions)

print(f'Test Valence MSE: {test_valence_mse:.4f}')
print(f'Test Arousal MSE: {test_arousal_mse:.4f}')
print(f'Test Emotion Accuracy: {test_emotion_accuracy:.4f}')
print(f'Test Valence CCC: {test_valenece_ccc:.4f}')