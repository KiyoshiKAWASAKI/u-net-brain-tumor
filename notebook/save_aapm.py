import numpy as np
import os, sys, glob
import numpy as np
import dicom
import matplotlib.pyplot as plt
%matplotlib inline
from skimage.draw import polygon
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Set the data path
train_image_path = "../../../data/AAPM/train_image/"
train_mask_path = "../../../data/AAPM/train_mask/"
valid_image_path = "../../../data/AAPM/valid_image/"
valid_mask_path = "../../../data/AAPM/valid_mask/"

train_patients = [os.path.join(train_image_path, name)
for name in os.listdir(train_image_path) if os.path.isdir(os.path.join(train_image_path, name))]

train_masks = [os.path.join(train_mask_path, name)
for name in os.listdir(train_mask_path) if os.path.isdir(os.path.join(train_mask_path, name))]

valid_patients = [os.path.join(valid_image_path, name)
for name in os.listdir(valid_image_path) if os.path.isdir(os.path.join(valid_image_path, name))]

valid_masks = [os.path.join(valid_mask_path, name)
for name in os.listdir(valid_mask_path) if os.path.isdir(os.path.join(valid_mask_path, name))]

# Save training images into a npy
import matplotlib.image as mpimg

train_image = []
train_mask = []

mask_base_path = "../../../data/AAPM/train_mask/"

for patient in train_patients:
    g = os.walk(patient)
    for path,dir_list,file_list in g:  
        for file_name in file_list: 
            full_path = os.path.join(path, file_name)
            
            patient_num = path[-3:]
            print "*Patient number: " + str(patient_num)
            file_num = int(file_name[:-4])+1
            file_num = str(file_num)
            print "  File number: " + str(file_num)
            txt_path = mask_base_path + patient_num + '/'+file_num+'.txt'
            m = np.loadtxt(txt_path)
            train_mask.append(m)
            
            
            img = mpimg.imread(full_path)
            img_array = np.array(img)
            new_img_array = img_array[:,:,0:3]
            gray_img = rgb2gray(new_img_array)
            
            print "  Shape: " + str(gray_img.shape)
            train_image.append(gray_img)
            
            #print(os.path.join(path, file_name))

np.save('../../../data/AAPM/npy_files/train_images', train_image)
np.save('../../../data/AAPM/npy_files/train_mask', train_mask)

# Saving validation in a numpy
valid_image = []
valid_mask = []

mask_base_path = "../../../data/AAPM/valid_mask/"

for patient in valid_patients:
    g = os.walk(patient)
    for path,dir_list,file_list in g:  
        for file_name in file_list: 
            full_path = os.path.join(path, file_name)
            
            patient_num = path[-3:]
            print "*Patient number: " + str(patient_num)
            file_num = int(file_name[:-4])+1
            file_num = str(file_num)
            print "  File number: " + str(file_num)
            txt_path = mask_base_path + patient_num + '/'+file_num+'.txt'
            m = np.loadtxt(txt_path)
            valid_mask.append(m)
            
            
            img = mpimg.imread(full_path)
            img_array = np.array(img)
            new_img_array = img_array[:,:,0:3]
            gray_img = rgb2gray(new_img_array)
            valid_image.append(gray_img)
            
            print"  Shape: " + str(gray_img.shape)

np.save('../../../data/AAPM/npy_files/valid_images', valid_image)
np.save('../../../data/AAPM/npy_files/valid_mask', valid_mask)