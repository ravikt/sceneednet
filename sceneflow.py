import re
import os
import numpy as np
import numpy.matlib
import sys
import matplotlib.pyplot as plt

import keras
from keras import backend as K 

class SceneFlow():

    def sceneflowconstruct(self,of, dt0, dt1):

        # Calculate depth using stereo baseline
        # Use camera matrix to get coordinates in the world frame
        # Use disparity change for sce flow
        # Plot three dimensional motion vectors 
        focal_length = 1050.0
        baseline = 1.0

        # Scene flow as 4D vector
        sf = np.dstack((of[:,:,0:2], dt0, dt1))

        row = np.arange(540)
        px  = np.transpose(np.matlib.repmat(row,960,1))

        column = np.ax = range(960)
        py = np.matlib.repmat(column,540, 1)

        px_offset = 0
        py_offset = 0
        u = of[:,:,0] # Optical flow in horizontal direction
        v = of[:,:,1] # optical flow in vertical direction

        z0 = (focal_length*baseline)/dt0
        x0 = np.multiply((px-px_offset),z0)/focal_length
        y0 = np.multiply((py-py_offset),z0)/focal_length

        print np.float32(x0).dtype

        z1 = (focal_length*baseline)/dt1
        x1 = np.multiply((px+u-px_offset),z1)/focal_length
        y1 = np.multiply((py+v-py_offset),z1)/focal_length


        # Scene flow vectors

        dX = np.float32(x1 - x0)
        dY = np.float32(y1 - y0)
        dZ = np.float32(z1 - z0)

        scene_flow = np.dstack((dX, dY, dZ))

        return scene_flow

    # End point error loss function
    def epeloss(self, y_true, y_pred):
        x = y_true[:,:,1] - y_pred[:,:,1]
        y = y_true[:,:,2] - y_pred[:,:,2]
        z = y_true[:,:,3] - y_pred[:,:,3]
        loss = K.square(x) + K.square(y) + K.square(z)
        loss = K.sqrt(loss)
        return loss 
