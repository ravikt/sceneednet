# SceneEDNet
# Fully convolutional neural network for scene flow estimation


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, \
     Cropping2D, Activation, Conv2DTranspose, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras import losses
from keras import optimizers
from keras.models import load_model

from keras.utils import multi_gpu_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import History

from sceneedclass import DataGenerator
from readfiles import dataRead
from sceneflow import sceneFlow

train_path = "training set path"
val_path = "validation set path"
Nepoch = 100

# total number of train and validation folders
train_steps = 'number of folders in training set'
val_steps  = 'number of folders in validation set'

# learning rate you want to keep 
learning_rate = 'learinng rate' 
#lrdecay = 0.0
lrdecay       = learning_rate/Nepoch

train_folder_list = range(0, train_steps)
val_folder_list   = range(0, val_steps)


# Data Generation for training and validation

training_generator   = DataGenerator().generate(train_path, train_folder_list)
validation_generator = DataGenerator().generate(val_path, val_folder_list) 
#test_generator     = DataGenerator().generate(test_folder_list)
print learning_rate


def sceneednet():

    input = Input(shape=(540, 960, 12))
    conv0 =   Conv2D(64,   (3, 3), name = 'conv0',   strides = 2, padding='same')(input)
    conv0 =   LeakyReLU()(conv0)
    conv1_1 = Conv2D(128,  (3, 3), name = 'conv1_1', strides = 2, padding='same')(conv0)
    conv1_1 = LeakyReLU()(conv1_1)
    conv3_1 = Conv2D(256,  (3, 3), name = 'conv3_1', strides = 2, padding='same')(conv1_1)
    conv3_1 = LeakyReLU()(conv3_1)
    conv5_1 = Conv2D(512,  (3, 3), name = 'conv5_1', strides = 2, padding='same')(conv3_1)
    conv5_1 = LeakyReLU()(conv5_1)
    conv6_1 = Conv2D(1024, (3, 3), name = 'conv6_1', strides = 1, padding='same')(conv5_1)
    conv6_1 = LeakyReLU()(conv6_1) 

    conv7_1 = Conv2D(1024, (3, 3), name = 'conv7_1', strides = 1, padding='same')(conv6_1)
    conv7_1 = LeakyReLU()(conv7_1)
    up1     = UpSampling2D((2,2))(conv7_1)
    conv8_1 = Conv2D(512, (3, 3), name = 'conv8_1', strides = 1, padding='same')(up1)
    conv8_1 = LeakyReLU()(conv8_1)
    up2     = UpSampling2D((2,2))(conv8_1)
    conv9_1 = Conv2D(256, (3, 3), name = 'conv9_1', strides = 1, padding='same')(up2)
    conv9_1 = LeakyReLU()(conv9_1)
    up3     = UpSampling2D((2,2))(conv9_1)
    conv10_1= Conv2D(128, (3, 3), name = 'conv10_1', strides = 1, padding='same')(up3)
    conv10_1= LeakyReLU()(conv10_1)
    up4     =  UpSampling2D((2,2))(conv10_1)
    out = Cropping2D(cropping=((4,0),(0,0)))(up4)
    conv11_1= Conv2D(64, (3, 3), name = 'conv11_1', strides = 1, padding='same')(out)
    conv11_1= LeakyReLU()(conv11_1)
    output = Conv2D(3, (3, 3), name = 'output', strides = 1, padding='same')(conv11_1)
    output = LeakyReLU()(output)
    model = Model(input=input, output=output)
    #model.compile(loss='mae', optimizer='adam')
    model.summary()
 
    return model


with tf.device('/cpu:0'):
    model = sceneednet()
#model = sceneednet()
parallel_model = multi_gpu_model(model, gpus=2)
'''
with tf.device('/cpu:0'):
     model = load_model('sceneflow.h5')
     #model = model.get_layer(name=model_1)

#model = load_model('sceneflowB.h5')
#keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)

# For fine tuning
parallel_model = multi_gpu_model(model, gpus=2)
'''
# Loss function and optimizer

adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
       epsilon=None, decay=lrdecay, amsgrad=False)

# Configure the model for training
parallel_model.compile(optimizer=adam, loss=sceneFlow().epeloss, metrics=['accuracy'])


# Saves the model after every epoch
checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=False)

# Print history to a file
csv_logger = keras.callbacks.CSVLogger('training.log')

# Tensorboard visulaization
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

# Stop the training if there is no improvement
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, 
            verbose=0, mode='auto')

# Training 
H = parallel_model.fit_generator(generator=training_generator, steps_per_epoch=train_steps, epochs=Nepoch, 
    validation_data=validation_generator, validation_steps=val_steps, callbacks=[csv_logger])

keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
#
keras.utils.print_summary(parallel_model, line_length=None, positions=None, print_fn=None)
#model.summary()
model.save('sceneflowA.h5')
