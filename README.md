
# SceneEDNet 
==================

Implementation of ICARCV 2018 version of [SceneEDNet](https://ieeexplore.ieee.org/abstract/document/8581172).

## Abstract
===================

Estimating scene flow in RGB-D videos is attracting much interest of the computer vision researchers, due to its potential applications in robotics. The state-ofthe-art techniques for scene flow estimation, typically rely on the knowledge of scene structure of the frame and the correspondence between frames. However, with the increasing amount of RGB-D data captured from sophisticated sensors like Microsoft Kinect, and the recent advances in the area of sophisticated deep learning techniques, introduction of an efficient deep learning technique for scene flow estimation, is becoming important. This paper introduces a first effort to apply a deep learning method for direct estimation of scene flow by presenting a fully convolutional neural network with an encoder-decoder (ED) architecture. The proposed network SceneEDNet involves estimation of three dimensional motion vectors of all the scene points from sequence of stereo images. The training for direct estimation of scene flow is done using consecutive pairs of stereo images and corresponding scene flow ground truth. The proposed architecture is applied on a huge dataset and provides meaningful results

## Dependencies
====================

Keras >= 2.1.4 

OpenCV


## Citation
====================

The code is available for research purpose. If you are using the code in your research work, please cite the following paper.

    @inproceedings{thakur2018sceneednet,
    title={SceneEDNet: A Deep Learning Approach for Scene Flow Estimation},
    author={Thakur, Ravi Kumar and Mukherjee, Snehasis},
    booktitle={2018 15th International Conference on Control, Automation, Robotics and Vision (ICARCV)},
    pages={394--399},
    year={2018},
    organization={IEEE}
    }
