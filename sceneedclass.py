import numpy as np
import cv2
import os
import re

from readfiles import DataRead
#from sceneflow import SceneFlow

class DataGenerator():

  def __data_generation(self, path, folder_number):

     xtrain = []
     ytrain = []
 
     img_dir = "/home/ravikumar/DataSets/frames_finalpass"
     of_dir  = "/home/ravikumar/DataSets/optical_flow"
     d_dir   = "/home/ravikumar/DataSets/disparity"

     focal_length = 1050.0
     baseline = 1.0

     
     for y in folder_number:
         
         if y<10:
       	    folder_path = "000%d" % (y)  
         elif y>=10 and y<100:
            folder_path = "00%d" % (y)
         elif y>=100 and y<1000:
            folder_path = "0%d" % (y)          
         else:
            folder_path = "%d" % (y)
            
         for x in range(6, 14):
            if x < 9:
                im_path1 = "000%d.png" % (x)
                im_path2 = "000%d.png" % (x+1)
                of_path = "OpticalFlowIntoFuture_000%d_L.pfm"  % (x+1)
                disp_path1 = "000%d.pfm"% (x)
                disp_path2 = "000%d.pfm"% (x+1)           

            elif x > 9:
                im_path1 = "00%d.png" % (x)
                im_path2 = "00%d.png" % (x+1)
                of_path = "OpticalFlowIntoFuture_00%d_L.pfm" % (x+1)
                disp_path1 = "00%d.pfm"%(x)
                disp_path2 = "00%d.pfm"%(x+1)  
   
            else:      # for x=9, gives frame 9 and 10
                im_path1 = "000%d.png" % (x)
                im_path2 = "00%d.png" % (x+1)
                of_path = "OpticalFlowIntoFuture_00%d_L.pfm" % (x+1)
                disp_path1 = "000%d.pfm"%(x)
                disp_path2 = "00%d.pfm"%(x+1)       
   
            imgL1 = cv2.imread(os.path.join(img_dir, path ,folder_path,"left", im_path1))
            imgR1 = cv2.imread(os.path.join(img_dir, path, folder_path,"right", im_path1)) # stereo pair at t
            imgL2 = cv2.imread(os.path.join(img_dir, path, folder_path,"left", im_path2))
            imgR2 = cv2.imread(os.path.join(img_dir, path, folder_path,"right", im_path2)) # stereo pair at t+1
              
            input_data = np.concatenate((imgL1, imgR1, imgL2, imgR2), axis = 2)

            #print y, x
            xtrain.append(input_data)
            
            of = dataRead().readPFM(os.path.join(of_dir, path, folder_path, "into_future", "left", of_path))
            disp1 = dataRead().readPFM(os.path.join(d_dir, path, folder_path, "left", disp_path1))
            disp2 = dataRead().readPFM(os.path.join(d_dir, path, folder_path, "left", disp_path2))
            
            target = dataRead().sceneflowconstruct(of, disp1, disp2)
            #depth1 = (focal_length*baseline)/disp1
            #depth2 = (focal_length*baseline)/disp2 
            
            #target = np.dstack((of, depth1, depth2))
            ytrain.append(target)

     xtrain = np.array(xtrain)
     ytrain = np.array(ytrain)
     #print y
     #print xtrain.shape, ytrain.shape
     return xtrain, ytrain

  def generate(self, path, folder_list):
     # Generate batches of samples
     while 1:
         # indexes = folder_list 
         # Take images from four folders
          imax = int(len(folder_list)/1) # 1,2,...number of folders
          for i in range(imax):
             folder_number = [folder_list[k] for k in folder_list[i*1:(i+1)*1]]
             #folder_number = folder_list[i]
             # Generate data       
             #print folder_number
             X, y = self.__data_generation(path, folder_number)
  
             yield X, y
