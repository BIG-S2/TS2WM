from __future__ import absolute_import
from __future__ import print_function
import os
import tensorflow as tf
import keras.models as models
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger, Callback
from keras.models import *
from loss import generalised_dice_loss,generalised_wasserstein_dice_loss
from keras import backend as K
from keras.utils import multi_gpu_model
from FC_densenet import FC_densenet
from batch_generator_step2_t1c import validate_generator,training_generator

import math
import cv2
import nibabel as nib
import numpy as np
import json
from random import randint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# This code is used for step1
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'
from keras.regularizers import l2

def soft_dice(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

num_classes=4
batch_size=2
nb_epoch=8
nb_epoch1=100
input_shape=(64,64,16,3)

model=FC_densenet(num_classes,8,input_shape)

sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)            
def step_decay(epoch):
    initial_lrate =0.01
    drop = 0.5
    epochs_drop =10
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

# validation and training subject name
temp1=np.load('/home/list1.npy')
temp2=np.load('/home/list2.npy')
##
def train_generator(batch_size):
    data,targets=training_generator(temp1)

    loopcount = data.shape[0] // batch_size
    while (True):
        i = randint(0,loopcount)

        X = data[i*batch_size : (i+1)*batch_size]
        Y =targets[i*batch_size : (i+1)*batch_size]
        yield (X, Y)
def val_generator(batch_size):
    data,targets=validate_generator(temp2)
    loopcount = data.shape[0] // batch_size
    while (True):
        i = randint(0,loopcount)

        X = data[i*batch_size : (i+1)*batch_size]
        Y =targets[i*batch_size : (i+1)*batch_size]
        yield (X, Y)

model.compile(loss=generalised_dice_loss, optimizer=sgd, metrics = ['categorical_accuracy',soft_dice])

saveBestModel=ModelCheckpoint("/home/step2/spdropout_fold_batch_t1c.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')

history=model.fit_generator(generator=train_generator(batch_size),
                    steps_per_epoch=25000/batch_size,
                    epochs=100,
                    validation_data=val_generator(batch_size),
                    validation_steps=5000/batch_size,
                    callbacks=[saveBestModel,lrate],
                    shuffle=True,
                    workers=4,
                    verbose = 2)
model.save("/home/step2/spdropout_fold_batch_t1c.h5")

with open('/home/step2/spdropout_fold_batch_t1c.txt','w') as f:
    f.write(str(history.history))
