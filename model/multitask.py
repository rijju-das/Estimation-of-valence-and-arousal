
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from skimage.transform import resize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import os
import cv2
import pandas as pd
import numpy as np

train_dir = 'train_set/images'
val_dir = 'val_set/images'
train_ann_file = 'train_annotation.csv'
val_ann_file = 'val_annotation.csv'
img_width=227
img_height=227
train_img=[]
val_img =[]
for f in os.listdir(train_dir):
    train_img.append(f)

for f in os.listdir(val_dir):
    val_img.append(f)

batch_size = 20
num_emotions = 8

# function to read the data from the directory
def one_hot_encode(number, num_classes=num_emotions):
    # Create an array of zeros with length equal to the number of categories
    one_hot_vector = np.zeros(num_classes)
    
    # Set the index corresponding to the number to 1
    one_hot_vector[number] = 1
    
    return one_hot_vector


def get_data(dir_path, ann_file, images, batch_size, dims):
    """
    Generates batches of images and corresponding labels.
    
    Parameters:
    dir_path (str): Directory where the actual images are kept.
    ann_file (str): Path to the annotation file (CSV).
    images (list): List of image filenames to generate batches from.
    batch_size (int): Number of images per batch.
    dims (list): Dimensions to rescale images to [height, width].
    
    Returns:
    tuple: Batch of images, arousal labels, valence labels, emotion labels.
    """
   
    while True:
        ix = np.random.choice(np.arange(len(images)), batch_size)
        imgs = []
        label_aro = []
        label_val = []
        label_emo = []

        for i in ix:
            # Load and preprocess image
            original_img = load_img(os.path.join(dir_path, images[i]))
            resized_img = resize(img_to_array(original_img), dims + [3])
            array_img = resized_img / 255.0
            imgs.append(array_img)

            # Load annotations
            img_name = images[i]
            anno = pd.read_csv(ann_file)
            # print(img_name.split(".")[0])
            row = anno[anno["filename"] == int(img_name.split(".")[0])]
            
            label_aro.append(row["Arousal"].values[0])
            # print(row["Arousal"].values[0])
            label_val.append(row["Valance"].values[0])
            label_emo.append(one_hot_encode(row["Expression"].values[0]))

        imgs = np.array(imgs)
        
        label_aro = np.array(label_aro).astype(np.float32)
        label_val = np.array(label_val).astype(np.float32)
        label_emo = np.array(label_emo).astype(np.float32)

        # print(images[i])
        
        # print(label_val)
        # print(label_emo)
        yield imgs, (label_val, label_aro, label_emo)


# # Define input shape
input_shape = [227, 227]

train_gen = get_data(train_dir, train_ann_file, train_img, batch_size, input_shape)

val_gen = get_data(val_dir, val_ann_file, val_img, batch_size, input_shape)

model = Sequential()

# Layer 1: Convolutional layer with 64 filters of size 11x11x3
model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu', input_shape=(227,227,3)))

# Layer 2: Max pooling layer with pool size of 3x3
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# Layer 3-5: 3 more convolutional layers with similar structure as Layer 1
model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# Layer 6: Fully connected layer with 6X6X256=9216 neurons
model.add(Flatten())
model.add(Dense(9216, activation='relu'))
model.add(Dense(256, activation='relu'))

# Layer 7: Another fully connected layer with 4096 neurons
model.add(Dense(256, activation='relu'))

# Get the output of the last Dense layer
flatten_common = model.output

# Define task-specific layers for valence
dense_valence = Dense(64, activation='relu')(flatten_common)
output_valence = Dense(1, activation='linear', name='valence')(dense_valence)

# Define task-specific layers for arousal
dense_arousal = Dense(64, activation='relu')(flatten_common)
output_arousal = Dense(1, activation='linear', name='arousal')(dense_arousal)

# Define task-specific layers for emotion

dense_emotion = Dense(64, activation='relu')(flatten_common)
output_emotion = Dense(num_emotions, activation='softmax', name='emotion')(dense_emotion)

# Define the model with multiple outputs
model = Model(inputs=model.input, outputs=[output_valence, output_arousal, output_emotion])

model.compile(optimizer=Adam(),
              loss={'valence': 'mse', 'arousal': 'mse', 'emotion': 'categorical_crossentropy'},
              metrics={'valence': ['mae'], 'arousal': ['mae'], 'emotion': ['accuracy']})

model.summary()

# Calculate steps per epoch for training and validation
epoch_steps = np.ceil(len(train_img) // batch_size).astype(int)
validation_steps = np.ceil(len(val_img) // batch_size).astype(int)
# Train the model using the generator

# history = model.fit(train_gen,validation_data = val_gen, steps_per_epoch=3, epochs=3,validation_steps=2, batch_size=batch_size)
history = model.fit(train_gen, steps_per_epoch=100, epochs=10, batch_size=batch_size)

val_loss, val_valence_loss, val_arousal_loss, val_emotion_loss, val_valence_mae, val_arousal_mae, val_emotion_accuracy = model.evaluate(val_gen, steps=100, verbose=0)

# Print evaluation metrics
print('Validation Loss: {:.4f}'.format(val_loss))
print('Validation Valence Loss: {:.4f}'.format(val_valence_loss))
print('Validation Arousal Loss: {:.4f}'.format(val_arousal_loss))
print('Validation Emotion Loss: {:.4f}'.format(val_emotion_loss))
print('Validation Valence MAE: {:.4f}'.format(val_valence_mae))
print('Validation Arousal MAE: {:.4f}'.format(val_arousal_mae))
print('Validation Emotion Accuracy: {:.4f}'.format(val_emotion_accuracy))
evaluation_metrics = {
    "Loss": round(val_loss,4),
    "Valence loss": round(val_valence_loss,4),
    "Arousal loss": round(val_arousal_loss,4),
    "Emotion loss": round(val_emotion_loss,4),
    "Valence mae": round(val_valence_mae,4),
    "Arousal mae": round(val_arousal_mae,4),
    "Emotion accuracy": round(val_emotion_accuracy,4)
}

# Convert the dictionary to a DataFrame
df_results = pd.DataFrame([evaluation_metrics])

df_results.to_csv("Results.csv")