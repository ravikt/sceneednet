# Code for generating predictions

import numpy as np
import cv2
import keras 
import matplotlib.pyplot as plt

from keras.models import load_model
from keras import backend as K
from keras import optimizers
from keras import losses

from readfiles import DataRead
from sceneflow import SceneFlow

# Use the saved model 
model = load_model('trained_model.h5')

# read set of stereo images at t and t+1
# total five sets from each folder

# dataset folder containing images with its 
# corresponding optical flow and disparity
predict_path = "data" 

input_images, ground_truth = DataRead().readInput(predict_path)
output = model.predict(input_images)

# extracting optical flow and depth maps from output
N = 1 # Sample number. There are 8 samples in a batch

predicted_sceneflow = output[N, :, :, :] # Sample number 

of_gt = ground_truth[N,:,:,0:3]  # ground truth optical flow
d1_gt = ground_truth[N,:,:,3]    # ground truth disparity for frames at t
d2_gt = ground_truth[N,:,:,4]    # ground truth disparity for frames at t+1

# Reconstructed scene flow gorund truth using optical flow and disparity
groundtruth_sceneflow = DataRead().sceneflowconstruct(of_gt, d1_gt, d2_gt) 



for i in range(3):

    plt.figure(figsize=[8,6])
    plt.axis('off')
    predFlow = predicted_sceneflow[:,:,i]
    plt.imshow(predFlow)
    # predFilename = "SF%dpr" % i
    # plt.title(predFilename, fontsize=16)
    plt.savefig('sf'+str(i)+'pr.png', transparent=True, bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=[8,6])
    plt.axis('off')
    gtFlow = groundtruth_sceneflow[:,:,i]
    plt.imshow(gtFlow)
    #gtFilename = "SF%dgt" % i
    # plt.title(gtFilename, fontsize=16)
    plt.savefig('sf'+str(i)+'gt.png', transparent=True, bbox_inches='tight') 
    plt.clf()



