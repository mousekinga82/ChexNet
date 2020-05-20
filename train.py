# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:52:01 2020

@author: mousekinga82
"""

import pandas as pd
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
import matplotlib.pyplot as plt

from utils import *
K.set_image_data_format('channels_last')

IMGDIR = './images'
IMG_H = 120
IMG_W = 120
batch_size = 1
LABELS = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 
          'Mass', 'Nodule', 'Atelectasis','Pneumothorax','Pleural_Thickening', 
          'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

#Load from csv
df = pd.read_csv('my_list.csv')
#leakage check
if leakage_check(df, 'Patient ID', 'Is_tv'):
    print('Leakage check pass :D')
else:
    print('Leakage check fail :(')
    exit()
    
np.random.seed(1)
#get mean and std of 1000 samples in train_val dataset
np.random.seed(1)
grab_mean_std = get_norm_data(df, image_dir = IMGDIR, H=IMG_H, W=IMG_W)
print(f'Sampled mean, std is : {grab_mean_std[0]:.2f}, {grab_mean_std[1]:.2f}')

#get generator for train & val set
train_gen, val_gen = get_tv_generator(df, grab_mean_std, LABELS, IMGDIR, IMG_H, IMG_W, batch_size = batch_size)
N_train, N_val, N_test = len(train_gen.filenames), len(val_gen.filenames), len(df[df['Is_tv'] == False])
print(f"train, val, test samples : {N_train}, {N_val}, {N_test}")

#Deal with class imbalance
plot_class_freq(LABELS, train_gen)
freq_pos, freq_neg = compute_class_freqs(train_gen)
plot_PN_class_ratio(train_gen, LABELS)

# create the base pre-trained model
base_model = DenseNet121(weights='imagenet', include_top=False)
x = base_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)
# and a logistic layer
predictions = Dense(len(LABELS), activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(freq_pos, freq_neg))
model.summary()

#history = 
model.fit_generator(train_gen, 
                              validation_data=val_gen,
                              steps_per_epoch=N_train/batch_size, 
                              validation_steps=N_val/batch_size, 
                              epochs = 3)

#plt.plot(history.history['loss'])
#plt.ylabel("loss")
#plt.xlabel("epoch")
#plt.title("Training Loss Curve")
#plt.show()