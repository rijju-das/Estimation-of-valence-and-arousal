import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.metrics import Precision, Recall
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_dir = 'train_set\\images'
val_dir = 'val_set\\images'
test_dir = 'test_set\\images'
train_ann_file = 'train_annotation.csv'
val_ann_file = 'val_annotation.csv'
img_width = 227
img_height = 227
train_img = []
val_img = []
test_img = []

for f in os.listdir(train_dir):
    train_img.append(f)

for f in os.listdir(val_dir):
    val_img.append(f)
    
for f in os.listdir(test_dir):
    test_img.append(f)

batch_size = 20
num_emotions = 8

# function to read the data from the directory
def one_hot_encode(number, num_classes=num_emotions):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[number] = 1
    return one_hot_vector

def data_generator(dir_path, ann_file, images, batch_size, input_shape, augment=False):
    # Define the ImageDataGenerator with or without augmentation
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
        # print(images)
        batch_paths = np.random.choice(images, size=batch_size)
        # print(batch_paths)
        batch_input = []
        batch_output_valence = []
        batch_output_arousal = []
        batch_output_emotion = []

        for input_path in batch_paths:
            img_path = os.path.join(dir_path, input_path)
            img = load_img(img_path, target_size=input_shape)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Reshape to add batch dimension
            if augment:
                img_array = next(datagen.flow(img_array, batch_size=1))[0]  # Apply augmentation and remove batch dimension
            else:
                img_array = datagen.standardize(img_array)  # Apply rescale preprocessing
                img_array = img_array[0]  # Remove batch dimension
            
            
            row = anno[anno["filename"] == int(input_path.split(".")[0])]
            if row.empty:
                print(f"Warning: No annotation found for {input_path}")
                continue
            batch_input.append(img_array)
            batch_output_valence.append(row["Valance"].values[0])
            batch_output_arousal.append(row["Arousal"].values[0])
            # batch_output_emotion.append(row["Expression"].values[0])
            batch_output_emotion.append(one_hot_encode(row["Expression"].values[0]))

        batch_input = np.array(batch_input)
        batch_output_valence = np.array(batch_output_valence)
        batch_output_arousal = np.array(batch_output_arousal)
        batch_output_emotion = np.array(batch_output_emotion)

        yield batch_input, {"valence": batch_output_valence, "arousal": batch_output_arousal, "emotion": batch_output_emotion}



# Define concordance_correlation_coefficient function
def CCC(y_true, y_pred):
    x = tf.convert_to_tensor(y_true)
    y = tf.convert_to_tensor(y_pred)
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.reduce_mean(xm * ym)
    r_den = tf.math.reduce_std(x) * tf.math.reduce_std(y)
    r = r_num / r_den
    ccc = 2 * r * tf.math.reduce_std(x) * tf.math.reduce_std(y) / (tf.math.reduce_variance(x) + tf.math.reduce_variance(y) + (mx - my)**2)
    return ccc


# Define the model using the Functional API
input_layer = Input(shape=(227, 227, 3))

# Layer 1: Convolutional layer with 64 filters of size 11x11x3
x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

# Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

# Layer 6: Fully connected layer with 9216 neurons
x = Flatten()(x)
x = Dense(9216, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Layer 7: Another fully connected layer with 4096 neurons
x = Dense(256, activation='relu')(x)

# Define task-specific layers for valence
dense_valence = Dense(64, activation='relu')(x)
output_valence = Dense(1, activation='sigmoid', name='valence')(dense_valence)

# Define task-specific layers for arousal
dense_arousal = Dense(64, activation='relu')(x)
output_arousal = Dense(1, activation='sigmoid', name='arousal')(dense_arousal)

# Define task-specific layers for emotion
dense_emotion = Dense(64, activation='relu')(x)
output_emotion = Dense(num_emotions, activation='softmax', name='emotion')(dense_emotion)

# Define the model with multiple outputs
model = Model(inputs=input_layer, outputs=[output_valence, output_arousal, output_emotion])

model.compile(optimizer=Adam(),
              loss={'valence': 'mse', 'arousal': 'mse', 'emotion': 'categorical_crossentropy'},
              metrics={'valence': ['mae', CCC], 'arousal': ['mae', CCC], 'emotion': ['accuracy', Precision(name='precision'), Recall(name='recall')]})

model.summary()

input_shape = [227, 227]
train_gen = data_generator(train_dir, train_ann_file, train_img, batch_size, input_shape, augment=True)
val_gen = data_generator(val_dir, val_ann_file, val_img, batch_size, input_shape, augment=True)
test_gen = data_generator(test_dir, val_ann_file, test_img, batch_size, input_shape, augment=True)


# Calculate steps per epoch for training and validation
epoch_steps = np.ceil(len(train_img) / batch_size).astype(int)
val_steps = np.ceil(len(val_img) / batch_size).astype(int)
test_steps = np.ceil(len(test_img) / batch_size).astype(int)


callbacks = [
    EarlyStopping(monitor='valence_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='valence_loss'),
    ReduceLROnPlateau(monitor='valence_loss', factor=0.2, patience=5, min_lr=1e-6)
]

history = model.fit(train_gen, steps_per_epoch=70, epochs=50,
                    validation_data=val_gen, validation_steps=50, callbacks=callbacks, verbose=1)

# Evaluate on test set
test_loss, test_valence_loss, test_arousal_loss, test_emotion_loss, test_valence_mae, test_valence_ccc, test_arousal_mae, test_arousal_ccc, test_emotion_accuracy, test_emotion_precision, test_emotion_recall = model.evaluate(test_gen, steps=5, verbose=0)

# Compute F1-score
test_emotion_f1_score = 2 * (test_emotion_precision * test_emotion_recall) / (test_emotion_precision + test_emotion_recall + tf.keras.backend.epsilon())

# Print evaluation metrics
print('Test Loss: {:.4f}'.format(test_loss))
print('Test Valence Loss: {:.4f}'.format(test_valence_loss))
print('Test Arousal Loss: {:.4f}'.format(test_arousal_loss))
print('Test Emotion Loss: {:.4f}'.format(test_emotion_loss))
print('Test Valence MAE: {:.4f}'.format(test_valence_mae))
print('Test Arousal MAE: {:.4f}'.format(test_arousal_mae))
print('Test Emotion Accuracy: {:.4f}'.format(test_emotion_accuracy))
print('Test Valence CCC: {:.4f}'.format(test_valence_ccc))
print('Test Arousal CCC: {:.4f}'.format(test_arousal_ccc))
print('Test Emotion Precision {:.4f}'.format(test_emotion_precision))
print('Test Emotion Recall {:.4f}'.format(test_emotion_recall))
print('Test Emotion Recall {:.4f}'.format(test_emotion_f1_score))


# Create a dictionary of the results and convert to DataFrame
evaluation_metrics = {
    "Loss": round(test_loss, 4),
    "Valence loss": round(test_valence_loss, 4),
    "Arousal loss": round(test_arousal_loss, 4),
    "Emotion loss": round(test_emotion_loss, 4),
    "Valence mae": round(test_valence_mae, 4),
    "Arousal mae": round(test_arousal_mae, 4),
    "Emotion accuracy": round(test_emotion_accuracy, 4),
    "Valence CCC": round(test_valence_ccc, 4),
    "Arousal CCC": round(test_arousal_ccc, 4),
    "Emotion Precision": round(test_emotion_precision, 4),
    "Emotion Recall": round(test_emotion_recall, 4),
    "Emotion F1 score": round(test_emotion_f1_score, 4),
}

df_results = pd.DataFrame([evaluation_metrics])

# Save the DataFrame to a CSV file
df_results.to_csv("Results.csv", index=False)
