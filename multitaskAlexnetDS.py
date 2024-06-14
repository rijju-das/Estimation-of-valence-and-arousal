import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import os

# Directory and file paths
train_dir = 'train_set\\images'
val_dir = 'val_set\\images'
train_ann_file = 'train_annotation.csv'
val_ann_file = 'val_annotation.csv'

# Parameters
input_shape = [227, 227]
batch_size = 20
num_emotions = 8

# Function to one-hot encode emotion labels
def one_hot_encode(number, num_classes=num_emotions):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[number] = 1
    return one_hot_vector

# Data generator function
def data_generator(dir_path, ann_file, batch_size, input_shape, augment=False):
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    anno = pd.read_csv(ann_file)
    while True:
        batch_paths = np.random.choice(anno['filename'], size=batch_size)
        batch_input, batch_output_valence, batch_output_arousal, batch_output_emotion = [], [], [], []

        for input_path in batch_paths:
            img_path = os.path.join(dir_path, str(input_path)+'.jpg')
            img = load_img(img_path, target_size=input_shape)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Reshape to add batch dimension
            if augment:
                img_array = next(datagen.flow(img_array, batch_size=1))[0]  # Apply augmentation and remove batch dimension
            else:
                img_array = datagen.standardize(img_array)  # Apply rescale preprocessing
                img_array = img_array[0]  # Remove batch dimension
            
            row = anno[anno["filename"] == input_path]
            if row.empty:
                print(f"Warning: No annotation found for {input_path}")
                continue
            batch_input.append(img_array)
            batch_output_valence.append(row["Valance"].values[0])
            batch_output_arousal.append(row["Arousal"].values[0])
            batch_output_emotion.append(one_hot_encode(row["Expression"].values[0]))

        batch_input = np.array(batch_input)
        batch_output_valence = np.array(batch_output_valence)
        batch_output_arousal = np.array(batch_output_arousal)
        batch_output_emotion = np.array(batch_output_emotion)

        yield batch_input, {"valence": batch_output_valence, "arousal": batch_output_arousal, "emotion": batch_output_emotion}

# Model definition
input_layer = Input(shape=(227, 227, 3))
x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(9216, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Valence task-specific layers
dense_valence = Dense(64, activation='relu')(x)
output_valence = Dense(1, activation='sigmoid', name='valence')(dense_valence)

# Arousal task-specific layers
dense_arousal = Dense(64, activation='relu')(x)
output_arousal = Dense(1, activation='sigmoid', name='arousal')(dense_arousal)

# Emotion task-specific layers
dense_emotion = Dense(64, activation='relu')(x)
output_emotion = Dense(num_emotions, activation='softmax', name='emotion')(dense_emotion)

# Compile the model
model = Model(inputs=input_layer, outputs=[output_valence, output_arousal, output_emotion])
model.compile(optimizer=Adam(),
              loss={'valence': 'mse', 'arousal': 'mse', 'emotion': 'categorical_crossentropy'},
              metrics={'valence': ['mae'], 'arousal': ['mae'], 'emotion': ['accuracy']})

# Print model summary
model.summary()

# Training and validation data generators
train_gen = data_generator(train_dir, train_ann_file, batch_size, input_shape, augment=True)
val_gen = data_generator(val_dir, val_ann_file, batch_size, input_shape, augment=False)

# Training parameters
epoch_steps = 30000 // batch_size
val_steps = 300

# Callbacks
callbacks = [
    EarlyStopping(monitor='valence_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='valence_loss'),
    ReduceLROnPlateau(monitor='valence_loss', factor=0.2, patience=5, min_lr=1e-6)
]

# Train the model
history = model.fit(train_gen, steps_per_epoch=epoch_steps, epochs=10, callbacks=callbacks)

# Collect true labels and predictions
y_true_valence, y_pred_valence = [], []
y_true_arousal, y_pred_arousal = [], []
y_true_emotion, y_pred_emotion = [], []

resultall = model.evaluate(val_gen, steps=val_steps)
for _ in range(val_steps):
    x_val, y_val = next(val_gen)
    y_pred = model.predict(x_val)
    
    y_true_valence.extend(y_val['valence'])
    y_pred_valence.extend(y_pred[0].squeeze())  # Squeeze to match shape
    y_true_arousal.extend(y_val['arousal'])
    y_pred_arousal.extend(y_pred[1].squeeze())  # Squeeze to match shape
    y_true_emotion.extend(y_val['emotion'])
    y_pred_emotion.extend(y_pred[2])

# Convert lists to arrays
y_true_valence = np.array(y_true_valence)
y_pred_valence = np.array(y_pred_valence)
y_true_arousal = np.array(y_true_arousal)
y_pred_arousal = np.array(y_pred_arousal)
y_true_emotion = np.argmax(np.array(y_true_emotion), axis=1)
y_pred_emotion = np.argmax(np.array(y_pred_emotion), axis=1)

# Calculate Precision, Recall, and F1-Score for emotion
true_positives = np.sum((y_pred_emotion == y_true_emotion) & (y_true_emotion == 1))
predicted_positives = np.sum(y_pred_emotion == 1)
actual_positives = np.sum(y_true_emotion == 1)

precision = true_positives / predicted_positives if predicted_positives > 0 else 0
recall = true_positives / actual_positives if actual_positives > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall + np.finfo(float).eps)

# Define CCC function
def ccc(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    covariance = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    true_var = np.mean((y_true - y_true_mean) ** 2)
    pred_var = np.mean((y_pred - y_pred_mean) ** 2)
    ccc_val = (2 * covariance) / (true_var + pred_var + (y_true_mean - y_pred_mean) ** 2)
    return ccc_val

# Calculate CCC for valence and arousal
ccc_valence = ccc(y_true_valence, y_pred_valence)
ccc_arousal = ccc(y_true_arousal, y_pred_arousal)

# Print evaluation metrics
print('Test Valence CCC: {:.4f}'.format(ccc_valence))
print('Test Arousal CCC: {:.4f}'.format(ccc_arousal))
print('Test Emotion Precision: {:.4f}'.format(precision))
print('Test Emotion Recall: {:.4f}'.format(recall))
print('Test Emotion F1-Score: {:.4f}'.format(f1_score))

# Create a dictionary of the results and convert to DataFrame
evaluation_metrics = {
    "Valence CCC": round(ccc_valence, 4),
    "Arousal CCC": round(ccc_arousal, 4),
    "Emotion Precision": round(precision, 4),
    "Emotion Recall": round(recall, 4),
    "Emotion F1-Score": round(f1_score, 4),
}

df_results = pd.DataFrame([evaluation_metrics])

# Save the DataFrame to a CSV file
df_results.to_csv("ResultsAlexNetDS.csv", index=False)
