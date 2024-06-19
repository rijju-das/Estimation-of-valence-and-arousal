


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

from skimage.transform import resize
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_dir = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/train_set/images'
val_dir = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/val_set/images'
train_ann_file = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/train_annotation.csv'
val_ann_file = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/val_annotation.csv'
img_width=224
img_height=224
train_img=[]
val_img =[]
for f in os.listdir(train_dir):
    train_img.append(f)

for f in os.listdir(val_dir):
    val_img.append(f)


num_emotions = 8

# function to read the data from the directory

class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class affectnet(Dataset):
    def __init__(self):
        
        imgs = []
        label_aro = []
        label_val = []
        label_emo = []
        train_dir = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/train_set/images'
        val_dir = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/val_set/images'
        train_ann_file = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/train_annotation.csv'
        val_ann_file = '/Users/rijju/Documents/Valence_arousal Work/AffectNet/val_annotation.csv'
        # cv2 - It reads in BGR format by default
        for f in os.listdir(val_dir):
            img = plt.imread(os.path.join(val_dir,f))
            # img = cv2.imread(f)
            # print(img)
            img = cv2.resize(img,(227,227)) # I can add this later in the boot-camp for more adventure
            b, g, r = cv2.split(img)
            img = cv2.merge([r,g,b])
            img = img.reshape((img.shape[2],img.shape[0],img.shape[1])) # otherwise the shape will be (h,w,#channels)
            imgs.append(img)

            anno = pd.read_csv(val_ann_file)
            # print(img_name.split(".")[0])
            row = anno[anno["filename"] == int(f.split(".")[0])]
            label_aro.append(row["Arousal"])
            label_val.append(row["Valance"])
            label_emo.append(row["Expression"])
            # label_emo.append(self.one_hot_encode(row["Expression"]))
            

        # our images
        self.images = np.array(imgs,dtype=np.float32)
        self.labels = np.array(label_val, dtype=np.float32)

    def one_hot_encode(number, num_classes=num_emotions):
        # Create an array of zeros with length equal to the number of categories
        one_hot_vector = np.zeros(num_classes)
        
        # Set the index corresponding to the number to 1
        one_hot_vector[number] = 1
    
        return one_hot_vector   
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        
        sample = {'image': self.images[index], 'label':self.labels[index]}
        
        return sample
    
    def normalize(self):
        self.images = self.images/255.0



class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)


device = torch.device('mps')
model = AlexNet().to(device)


dataset = affectnet()
dataset.normalize()
eta = 0.0001
EPOCH = 400
optimizer = torch.optim.Adam(model.parameters(), lr=eta)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(1, EPOCH):
    losses = []
    for D in dataloader:
        optimizer.zero_grad()
        data = D['image'].to(device)
        label = D['label'].to(device)
        y_hat = model(data)
        # define loss function
        error = nn.BCELoss() 
        loss = torch.sum(error(y_hat.squeeze(), label))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch+1, np.mean(losses)))