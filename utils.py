# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:57:28 2020

@author: mousekinga82
"""
import numpy as np
import os
import pandas as pd
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns

def leakage_check(df, id_col:str, is_tv_col:str):
    set1 = set(df[df[is_tv_col] == True][id_col].values)
    set2 = set(df[df[is_tv_col] == False][id_col].values)
    set_both = set1.intersection(set2)
    if_leak = len(set_both) == 0
    return if_leak

#for data normalization
def get_norm_data(df, image_dir, sample_size = 1000, img_col = 'Image', is_tv_col = 'Is_tv', H=240, W=240):
    #np.random.seed(0)
    imgs_list = np.random.choice(df[df[is_tv_col] == True][img_col].values, sample_size)
    grab_list = []
    for img in imgs_list:
        grab_list.append(
            np.array(image.load_img(os.path.join(image_dir, img), target_size = (H,W))))
    return np.mean(grab_list), np.std(grab_list)

def get_tv_generator(df, grab_mean_std, labels, img_dir, H, W, img_col ='Image', is_tv_col = 'Is_tv', shuffle=True, batch_size=8, seed = 1):
    image_train_gen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True,
        rotation_range = 100,
        zoom_range = 0.1,
        horizontal_flip = True,
        validation_split = 0.2)
    image_train_gen.mean, image_train_gen.std = grab_mean_std
    
    train_gen = image_train_gen.flow_from_dataframe(
        dataframe = df[df['Is_tv'] == True],
        directory = img_dir,
        x_col = img_col,
        y_col = labels,
        class_mode = 'raw',
        batch_size =  batch_size,
        shuffle = shuffle,
        seed = seed,
        target_size = (W, H),
        subset = 'training')
    
    image_val_gen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True)
    image_val_gen.mean, image_val_gen.std = grab_mean_std
    
    val_gen = image_val_gen.flow_from_dataframe(
        dataframe = df[df['Is_tv']==True & ~df['Image'].isin(train_gen.filenames)],
        directory = img_dir,
        x_col = img_col,
        y_col = labels,
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        seed = seed,
        target_size = (W, H))
    return train_gen, val_gen

def plot_class_freq(labels, train_gen):
    plt.xticks(rotation=90)
    plt.bar(x=labels, height=np.sum(train_gen.labels, axis=0))
    plt.title("Frequency of Each Class")
    plt.show()
    
def compute_class_freqs(gen):
    """
    Compute positive and negative frequences for each class in a generator.

    Args:
        gen (data generator) 
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    labels = gen.labels
    N = labels.shape[0]
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = np.sum(1 - labels, axis=0) / N
    return positive_frequencies, negative_frequencies

def plot_PN_class_ratio(gen, labels):
    freq_pos, freq_neg = compute_class_freqs(gen)
    data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
    data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
    plt.xticks(rotation=90)
    f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

def get_weighted_loss(freq_pos, freq_neg, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      freq_pos (np.array): array of positive weights for each class, size (num_classes)
      freq_neg (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        for i in range(len(freq_pos)):
            # for each class, add average weighted loss for that class 
            loss += -K.mean(freq_pos[i]*y_true[:,i]*K.log(y_pred[:,i] + epsilon) + freq_neg[i]*(1-y_true[:,i])*K.log(1-y_pred[:,i] + epsilon))   #complete this line
        return loss
    
    return weighted_loss