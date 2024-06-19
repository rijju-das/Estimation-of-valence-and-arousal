import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D
from tensorflow.keras import Model
train_dir = 'train_set\\images'
val_dir = 'val_set\\images'

train_ann_file = 'train_annotation.csv'
val_ann_file = 'val_annotation.csv'
img_width = 224
img_height = 224
train_img = []
val_img = []


for f in os.listdir(train_dir):
    train_img.append(f)

for f in os.listdir(val_dir):
    val_img.append(f)
    


batch_size = 20
num_emotions = 8

# function to read the data from the directory
def one_hot_encode(number, num_classes=num_emotions):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[number] = 1
    return one_hot_vector

def get_data(dir_path, ann_file, images, batch_size, dims):
    while True:
        ix = random.sample(images, batch_size)
        
        imgs = []
        label_aro = []
        label_val = []
        label_emo = []
        # print(ix)
        for i in ix:
            original_img = load_img(os.path.join(dir_path, i))
            resized_img = resize(img_to_array(original_img), dims + [3])
            array_img = resized_img / 255.0
            imgs.append(array_img)

            anno = pd.read_csv(ann_file)
            row = anno[anno["filename"] == int(i.split(".")[0])]

            label_aro.append(row["Arousal"].values[0])
            label_val.append(row["Valance"].values[0])
            label_emo.append(row["Expression"].values[0])
            # label_emo.append(one_hot_encode(row["Expression"].values[0]))

            # print(i)
            # print(label_val[-1])
            # print(label_aro[-1])

        imgs = np.array(imgs)
        label_aro = np.array(label_aro).astype(np.float32)
        label_val = np.array(label_val).astype(np.float32)
        label_emo = np.array(label_emo)
        
        

        yield imgs, (label_val, label_aro, label_emo)



def data_generator(dir_path, ann_file, images, batch_size, input_shape):
    datagen = ImageDataGenerator(rescale=1./255)
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
            img_array = datagen.standardize(img_array) # Apply rescale preprocessing
            img_array = img_array[0] # Remove batch dimension
            anno = pd.read_csv(ann_file)
            row = anno[anno["filename"] == int(input_path.split(".")[0])]

            batch_input.append(img_array)
            # print(row["Valance"])
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


# MobileNet block
def mobilnet_block (x, filters, strides):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x
#stem of the model
input_layer = Input(shape = (224,224,3))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = ReLU()(x)
# main part of the model
x = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters = 128, strides = 2)
x = mobilnet_block(x, filters = 128, strides = 1)
x = mobilnet_block(x, filters = 256, strides = 2)
x = mobilnet_block(x, filters = 256, strides = 1)
x = mobilnet_block(x, filters = 512, strides = 2)
for _ in range (5):
     x = mobilnet_block(x, filters = 512, strides = 1)
x = mobilnet_block(x, filters = 1024, strides = 2)
x = mobilnet_block(x, filters = 1024, strides = 1)
x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)


x = Flatten()(x)
x = Dense(1000, activation='relu')(x)


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
              metrics={'valence': ['mae', CCC], 'arousal': ['mae', CCC], 'emotion': ['accuracy']})

# model.compile(optimizer=Adam(),
#               loss={'valence': 'mse', 'arousal': 'mse', 'emotion': 'sparse_categorical_crossentropy'},
#               metrics={'valence': ['mae', CCC], 'arousal': ['mae', CCC], 'emotion': ['accuracy']})


model.summary()
#plot the model
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=False,show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

input_shape = [224, 224]
# train_gen = get_data(train_dir, train_ann_file, train_img, batch_size, input_shape)
# val_gen = get_data(val_dir, val_ann_file, val_img, batch_size, input_shape)
# test_gen = get_data(test_dir, val_ann_file, test_img, batch_size, input_shape)

train_gen = data_generator(train_dir, train_ann_file, train_img, batch_size, input_shape)
val_gen = data_generator(val_dir, val_ann_file, val_img, batch_size, input_shape)



# Calculate steps per epoch for training and validation
epoch_steps = np.ceil(3000/ batch_size).astype(int)
val_steps = np.ceil(len(val_img) / batch_size).astype(int)



callbacks = [
    EarlyStopping(monitor='valence_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='valence_loss'),
    ReduceLROnPlateau(monitor='valence_loss', factor=0.2, patience=5, min_lr=1e-6)
]

history = model.fit(train_gen, steps_per_epoch=30, epochs=1, callbacks=callbacks, verbose=1)

# Evaluate on test set
test_loss, test_valence_loss, test_arousal_loss, test_emotion_loss, test_valence_mae, test_arousal_mae, test_emotion_accuracy, test_valence_ccc, test_arousal_ccc = model.evaluate(val_gen, steps=5, verbose=0)

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
    "Arousal CCC": round(test_arousal_ccc, 4)
}

df_results = pd.DataFrame([evaluation_metrics])

# Save the DataFrame to a CSV file
df_results.to_csv("ResultsMobileNet.csv", index=False)
