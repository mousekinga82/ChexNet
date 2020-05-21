# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:52:01 2020

@author: mousekinga82
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from datetime import date
from utils import *
K.set_image_data_format('channels_last')

IMGDIR = './images'
IMG_H = 180
IMG_W = 180
batch_size = 64
val_split = 0.2

#Load from csv
df = pd.read_csv('my_list.csv')
df_key = list(df.keys())
LABELS = [e for e in df_key if e not in ('Patient ID', 'Patient Age', 'Patient Gender', 'Image', 'Is_tv')]

#leakage check
if leakage_check(df, 'Patient ID', 'Is_tv'):
    print('Leakage check pass :D')
else:
    print('Leakage check fail :(')
    exit()
    
#get mean and std of 1000 samples in train_val dataset
np.random.seed(1)
grab_mean_std = get_norm_data(df, image_dir = IMGDIR, H=IMG_H, W=IMG_W)
print(f'Sampled mean, std is : {grab_mean_std[0]:.2f}, {grab_mean_std[1]:.2f}')

#get split array for coss-validation
folds, most_n = get_split(df, val_split)
print(f"Validation Splits = {val_split}, Available folds for validation : {most_n}")
train_fold, val_fold = get_train_val_split(folds, 0)

#get generator for train & val set
train_gen, val_gen = get_tv_generator(df, train_fold, val_fold, grab_mean_std, LABELS, IMGDIR, IMG_H, IMG_W, batch_size = batch_size)
N_train, N_val, N_test = len(train_gen.filenames), len(val_gen.filenames), len(df[df['Is_tv'] == False])
print(f"train, val, test samples : {N_train}, {N_val}, {N_test}")

#Deal with class imbalance
plot_class_freq(LABELS, train_gen)
freq_pos, freq_neg = compute_class_freqs(train_gen)
plot_PN_class_ratio(train_gen, LABELS)

# create the base pre-trained model
base_model = DenseNet121(weights='densenet.hdf5', include_top=False)
x = base_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)
# and a logistic layer
predictions = Dense(len(LABELS), activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

#complie the model
model.compile(optimizer=Adam(learning_rate=0.01), loss=get_weighted_loss(freq_pos, freq_neg), metrics=[tf.keras.metrics.AUC()])

#define callbacks
day = str(date.today())
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
             ModelCheckpoint(f'Model_Weights_{day}.h5', save_best_only=True, save_weights_only=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-8),
             CSVLogger(f'log_{day}.csv', separator=",", append=False)]

#start training
model.fit_generator(train_gen, 
                    validation_data=val_gen,
                    steps_per_epoch=N_train/batch_size, 
                    validation_steps=N_val/batch_size, 
                    epochs = 100,
                    callbacks = callbacks)

model.load_weights('Model_Weights_2020-05-21.h5')

#Testing
test_gen = get_test_generator(df, grab_mean_std, LABELS, IMGDIR, IMG_H, IMG_W)
predicted_vals = model.predict(test_gen, steps = len(test_gen))

#ROC curve
auc_rocs = get_roc_curve(LABELS, predicted_vals, test_gen)
#GradCAM, only show the lables with top 4 AUC
labels_to_show = np.take(LABELS, np.argsort(auc_rocs)[::-1])[:4]
compute_gradcam(model, grab_mean_std, '00000096_001.png', IMGDIR, df, LABELS, labels_to_show)    