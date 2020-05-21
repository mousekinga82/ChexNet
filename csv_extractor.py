# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:03:15 2020

@author: mousekinga82
"""
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm

train_val_txt = 'train_val_list.txt'
test_txt = 'test_list.txt'
total_csv = 'Data_Entry_2017_v2020.csv'
imgs_dir = './images'
save_csv = 'my_list2.csv'

#Sort the current sample into train_val/test by text file
my_list = []
for i in glob.glob(imgs_dir + '/*.png'):
    my_list.append((i.replace(imgs_dir + '\\', '')))
    
total_df = pd.read_csv(total_csv)

#label of pathology
L = len(my_list)
arr_Atelectasis = np.zeros(L, dtype=bool)
arr_Cardiomegaly = np.zeros(L, dtype=bool)
arr_Consolidation = np.zeros(L, dtype=bool)
arr_Edema = np.zeros(L, dtype=bool)
arr_Effusion = np.zeros(L, dtype=bool)
arr_Emphysema = np.zeros(L, dtype=bool)
arr_Fibrosis = np.zeros(L, dtype=bool) 
arr_Hernia = np.zeros(L, dtype=bool)
arr_Infiltration = np.zeros(L, dtype=bool)
arr_Mass = np.zeros(L, dtype=bool)
arr_Nodule = np.zeros(L, dtype=bool)
arr_Pleural_Thickening = np.zeros(L, dtype=bool)
arr_Pneumonia = np.zeros(L, dtype=bool)
arr_Pneumothorax = np.zeros(L, dtype=bool)
#non-pathology info
arr_Image = [""] * L
arr_PatientID = np.zeros(L, dtype=np.uint32)
arr_PatientAge = np.zeros(L, dtype=np.uint8)
arr_PatientGender = [""] * L
#separate train_val / test
arr_is_tv = np.zeros(L, dtype=bool)

#dict_label = {'Atelectasis':arr_Atelectasis, 'Cardiomegaly':arr_Cardiomegaly, 'Consolidation':arr_Consolidation,
#              'Edema':arr_Edema, 'Effusion':arr_Effusion, 'Emphysema':arr_Emphysema, 'Fibrosis':arr_Fibrosis,
#              'Hernia':arr_Hernia, 'Infiltration':arr_Infiltration, 'Mass':arr_Mass, 'Nodule':arr_Nodule,
#              'Pleural_Thickening':arr_Pleural_Thickening, 'Pneumonia':arr_Pneumonia, 'Pneumothorax':arr_Pneumothorax,
#              'Patient ID':arr_PatientID, 'Patient Age':arr_PatientAge, 'Patient Gender':arr_PatientGender,
#              'Image': arr_Image, 'Is_tv':arr_is_tv}

dict_label = {'Cardiomegaly':arr_Cardiomegaly, 
              'Emphysema':arr_Emphysema, 
              'Effusion':arr_Effusion, 
              'Hernia':arr_Hernia, 
              'Infiltration':arr_Infiltration, 
              'Mass':arr_Mass, 
              'Nodule':arr_Nodule, 
              'Atelectasis':arr_Atelectasis,
              'Pneumothorax':arr_Pneumothorax,
              'Pleural_Thickening':arr_Pleural_Thickening, 
              'Pneumonia':arr_Pneumonia, 
              'Fibrosis':arr_Fibrosis, 
              'Edema':arr_Edema, 
              'Consolidation':arr_Consolidation,
              'Patient ID':arr_PatientID,
              'Patient Age':arr_PatientAge,
              'Patient Gender':arr_PatientGender,
              'Image':arr_Image,
              'Is_tv':arr_is_tv}

#Initialization
tmp_df = None
tmp_lables = None
tv_list, test_list = [], []
with open(train_val_txt, 'r') as f:
    for line in f:
        tv_list.append(line[:16])
with open(test_txt, 'r') as f:
    for line in f:
        test_list.append(line[:16])
#Use set for faster comparison
tv_set = set(tv_list)
test_set = set(test_list) 

#Compare the csv file
for index, imgs in enumerate(tqdm(my_list, desc='Comparing csv file')):
    #print(index, imgs)
    tmp_df = total_df[total_df['Image Index'] == imgs]
    if(len(tmp_df) == 0): 
        print('Image not found in the list !')
        print(index, imgs)
        break
    elif(len(tmp_df) > 1): 
        print('More than one image found in the list !')
        break
    else:
        tmp_labels = tmp_df['Finding Labels'].values[0].split('|')
        if tmp_df['Finding Labels'].values != 'No Finding':
            #fill in pathology labels
            for tmp_label in tmp_labels:
                dict_label[tmp_label][index] = 1
        #Fill the non-pathology info
        dict_label['Image'][index] = tmp_df['Image Index'].values[0]
        dict_label['Patient ID'][index] = tmp_df['Patient ID'].values[0]
        dict_label['Patient Age'][index] = tmp_df['Patient Age'].values[0]
        dict_label['Patient Gender'][index] = tmp_df['Patient Gender'].values[0]
        if (tmp_df['Image Index'].values[0] in tv_set):
            dict_label['Is_tv'][index] = 1
        elif((tmp_df['Image Index'].values[0] in test_set)):
            pass
        else:
            print('Image not found in train_val & test list !')
            break
#Data leakage check
    

#Save File            
df = pd.DataFrame(data = dict_label)
df.to_csv(save_csv, index=False)
print('Save file: ',save_csv)