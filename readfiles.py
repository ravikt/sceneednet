import cv2
import numpy as np
import os
import re
import numpy.matlib
import sys
import matplotlib.pyplot as plt

class dataRead():

    def readInput(self, path):
           
        data = []
        ground_truth=[]
        focal_length = 1050.0
        baseline = 1.0

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
               
            imgL1 = cv2.imread(os.path.join(path, "images/left", im_path1))
            imgR1 = cv2.imread(os.path.join(path, "images/right",im_path2)) # stereo pair at t
            imgL2 = cv2.imread(os.path.join(path, "images/left", im_path1))
            imgR2 = cv2.imread(os.path.join(path, "images/right", im_path2)) # stereo pair at t+1
 
            images = np.concatenate((imgL1, imgR1, imgL2, imgR2), axis = 2)
            
            data.append(images)
            # Ground truth optical flow and disparity

            of = dataRead().readPFM(os.path.join(path,"optical_flow/into_future/left", of_path))
            disp1 = dataRead().readPFM(os.path.join(path,"disparity/left", disp_path1))
            disp2 = dataRead().readPFM(os.path.join(path,"disparity/left", disp_path2))
            
            #depth1 = (focal_length*baseline)/disp1
            #depth2 = (focal_length*baseline)/disp2 
            target = np.dstack((of, disp1, disp2))
            ground_truth.append(target)

        data = np.array(data)
        ground_truth = np.array(ground_truth)
        return data, ground_truth

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

        px_offset = 479.5
        py_offset = 269.5
        u = of[:,:,0] # Optical flow in horizontal direction
        v = of[:,:,1] # optical flow in vertical direction

        z0 = (focal_length*baseline)/dt0
        x0 = np.multiply((px-px_offset),z0)/focal_length
        y0 = np.multiply((py-py_offset),z0)/focal_length

        #print np.float32(x0).dtype

        z1 = (focal_length*baseline)/dt1
        x1 = np.multiply((px+u-px_offset),z1)/focal_length
        y1 = np.multiply((py+v-py_offset),z1)/focal_length


        # Scene flow vectors

        dX = np.float32(x1 - x0)
        dY = np.float32(y1 - y0)
        dZ = np.float32(z1 - z0)

        scene_flow = np.dstack((dX, dY, dZ))

        return scene_flow

    # For reading and writing PFM file. 
    # Source https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    def writePFM(self,file, image, scale=1):
        
        file = open(file, 'wb')

        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')
      
        image = np.flipud(image)  

        if len(image.shape) == 3 and image.shape[2] == 3: # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        file.write('PF\n' if color else 'Pf\n')
        file.write('%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        file.write('%f\n' % scale)

        image.tofile(file)


    def readPFM(self, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data
    
