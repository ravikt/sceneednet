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
model = load_model('sceneflow.h5')
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])

# read set of stereo images at t and t+1
# total five sets from each folder

predict_path = "path to ground truth images, disparity and optical flow"
input_images, ground_truth = dataRead().readInput(predict_path)
output = model.predict(input_images)

#print output.shape, ground_truth.shape

# extracting optical flow and depth maps from output
N = 1 # Sample number. There are 8 samples in a batch

pair = output[N, :, :, :] # Sample number 


dx = pair[:,:,0]
dy = pair[:,:,1]
dz = pair[:,:,2]
#print optical_flow_pr[:,:,2]

# Writing the output in pfm files

plt.figure(figsize=[8,6])
plt.imshow(dx)
plt.title('dX(Pr)', fontsize=16)

plt.figure(figsize=[8,6])
plt.imshow(dy)
plt.title('dY(Pr)', fontsize=16)

plt.figure(figsize=[8,6])
plt.imshow(dz)
plt.title('dZ(Pr)', fontsize=16)


# Display ground truth


of_gt = ground_truth[N,:,:,0:3]
d1_gt = ground_truth[N,:,:,3]
d2_gt = ground_truth[N,:,:,4]
sfgt = dataRead().sceneflowconstruct(of_gt, d1_gt, d2_gt)

plt.figure(figsize=[8,6])
plt.imshow(sfgt[:,:,0])
plt.title('dX(GT)', fontsize=16)

plt.figure(figsize=[8,6])
plt.imshow(sfgt[:,:,1])
plt.title('dY(GT)', fontsize=16)

plt.figure(figsize=[8,6])
plt.imshow(sfgt[:,:,2])
plt.title('dZ(GT)', fontsize=16)

plt.show()
