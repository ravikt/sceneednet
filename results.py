# Code for generating predictions

import numpy as np
import cv2
import keras 
import matplotlib.pyplot as plt

from keras.models import load_model
from readfiles  import dataRead
from keras import optimizers
from keras import losses

# Use the saved model 
model = load_model('trained_model.h5')

# read set of stereo images at t and t+1
# total five sets from each folder

# dataset folder containing images with its 
# corresponding optical flow and disparity
predict_path = "data" 

input_images, ground_truth = dataRead().readInput(predict_path)
output = model.predict(input_images)

# extracting optical flow and depth maps from output
N = 1 # Sample number. There are 8 samples in a batch

pair = output[N, :, :, :] # Sample number 

of_gt = ground_truth[N,:,:,0:3]  # ground truth optical flow
d1_gt = ground_truth[N,:,:,3]    # ground truth disparity for frames at t
d2_gt = ground_truth[N,:,:,4]    # ground truth disparity for frames at t+1

# Reconstructed scene flow gorund truth using optical flow and disparity
sfgt = DataRead().sceneflowconstruct(of_gt, d1_gt, d2_gt) 


for i in range(3):

    plt.figure(figsize=[8,6])
    plt.axis('off')
    plt.imshow(pair[:,:,i])
    # predFilename = "SF%dpr" % i
    # plt.title(predFilename, fontsize=16)
    plt.savefig('sf'+str(i)+'pr.png', bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=[8,6])
    plt.axis('off')
    plt.imshow(sfgt[:,:,i])
    #gtFilename = "SF%dgt" % i
    # plt.title(gtFilename, fontsize=16)
    plt.savefig('sf'+str(i)+'gt.png', bbox_inches='tight') 
    plt.clf()
