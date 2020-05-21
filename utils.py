# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:57:28 2020

@author: mousekinga82
"""
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensoflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import cv2

def leakage_check(df, id_col:str, is_tv_col:str):
    set1 = set(df[df[is_tv_col] == True][id_col].values)
    set2 = set(df[df[is_tv_col] == False][id_col].values)
    set_both = set1.intersection(set2)
    if_leak = len(set_both) == 0
    return if_leak

def get_split(df, val_split, is_tv_col = 'Is_tv'):
    #how many data in a fold
    n = int(len(df[df[is_tv_col] == True]) // ( 1 / val_split))
    #random suffle the train_val set
    tv_list = df[df['Is_tv']==True]['Image'].values
    np.random.shuffle(tv_list)
    #divide into train, val
    folds = [ tv_list[i:i+n] for i in range(0, len(tv_list), n) ]
    return folds, int(len(df[df[is_tv_col] == True])/n)

def get_train_val_split(folds, val_index):
    train_fold = np.array([])
    for i, f in enumerate(folds):
        if i == val_index : continue
        else:
            train_fold= np.concatenate((train_fold, f))
    return train_fold, folds[val_index]

#for data normalization
def get_norm_data(df, image_dir, sample_size = 1000, img_col = 'Image', is_tv_col = 'Is_tv', H=240, W=240):
    #np.random.seed(0)
    imgs_list = np.random.choice(df[df[is_tv_col] == True][img_col].values, sample_size)
    grab_list = []
    for img in imgs_list:
        grab_list.append(
            np.array(image.load_img(os.path.join(image_dir, img), target_size = (H,W))))
    return np.mean(grab_list), np.std(grab_list)

def get_tv_generator(df, fold_train, fold_val, grab_mean_std, labels, img_dir, H, W, img_col ='Image', shuffle=True, batch_size=8, seed = 1):
    image_train_gen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True,
        rotation_range = 7,
        zoom_range = 0.1,
        horizontal_flip = True,
        validation_split = 0.)
    image_train_gen.mean, image_train_gen.std = grab_mean_std
    
    train_gen = image_train_gen.flow_from_dataframe(
        dataframe = df[df['Image'].isin(fold_train)],
        directory = img_dir,
        x_col = img_col,
        y_col = labels,
        class_mode = 'raw',
        batch_size =  batch_size,
        shuffle = shuffle,
        seed = seed,
        target_size = (W, H))
    
    image_val_gen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True)
    image_val_gen.mean, image_val_gen.std = grab_mean_std
    
    val_gen = image_val_gen.flow_from_dataframe(
        dataframe = df[df['Image'].isin(fold_val)],
        directory = img_dir,
        x_col = img_col,
        y_col = labels,
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        seed = seed,
        target_size = (W, H))
    return train_gen, val_gen

def get_test_generator(df, grab_mean_std, labels, img_dir, H, W, img_col = 'Image', is_tv_col = 'Is_tv', batch_size=8, seed = 1):
    image_test_gen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True)
    image_test_gen.mean, image_test_gen.std = grab_mean_std
    
    test_gen = image_test_gen.flow_from_dataframe(
        dataframe = df[df[is_tv_col] == False],
        directory = img_dir,
        x_col = img_col,
        y_col = labels,
        class_mode = 'raw',
        batch_size = batch_size,
        shuffle = False,
        seed = seed,
        target_size = (W, H))
    return test_gen

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
            loss += -K.mean(freq_neg[i]*y_true[:,i]*K.log(y_pred[:,i] + epsilon) + freq_pos[i]*(1-y_true[:,i])*K.log(1-y_pred[:,i] + epsilon))   #complete this line
        return loss
    
    return weighted_loss

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals

def load_image(img, image_dir, grab_mean_std, preprocess=True, H=180, W=180):
    """Load and preprocess image."""
    img_path = os.path.join(image_dir,img)
    mean, std = grab_mean_std
    x = image.load_img(img_path, target_size=(H, W))
    x = np.array(x)
    if preprocess:
        x = x - mean
        x = x / std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(input_model, image, clss, layer_name, H=180, W=180):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, clss]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, grab_mean_std, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, grab_mean_std)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, grab_mean_std, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, grab_mean_std, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1
