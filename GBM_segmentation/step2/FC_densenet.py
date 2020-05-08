import os
import tensorflow as tf
import keras.models as models
from keras.layers import Input,concatenate
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Input, add, concatenate, Conv3D, MaxPooling3D, UpSampling3D,SpatialDropout3D, Lambda, Activation,Deconvolution3D
from keras.layers.normalization import BatchNormalization
from functools import partial
from keras.layers.advanced_activations import PReLU
from keras.models import *
from keras import backend as K
data_format="channels_last"

def dense_block(filters,D1):
    channel_axis=-1
    DB2=BatchNormalization(axis=channel_axis)(D1)
    DB2=PReLU()(DB2)
    DB2=Conv3D(filters, 3,padding='same',kernel_initializer='he_normal')(DB2)
    DB2=SpatialDropout3D(rate=0.2, data_format=data_format)(DB2)
    DB2=concatenate([D1,DB2],channel_axis)
    for i in range(3): #block1 4*12=48
       DB1=BatchNormalization(axis=channel_axis)(DB2)
       DB1=PReLU()(DB1)
       DB1=Conv3D(filters, 3, padding='same',kernel_initializer='he_normal')(DB1)
       DB1=SpatialDropout3D(rate=0.2, data_format=data_format)(DB1)
       DB2=concatenate([DB2,DB1],channel_axis)
       
    return DB2
sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)

def FC_densenet(num_classes,filters,input_shape):
    inputs2=Input(input_shape)
    channel_axis=-1
    conv1=Conv3D(16,3,padding='same',kernel_initializer='he_normal')(inputs2)
    conv1=BatchNormalization(axis=channel_axis)(conv1)
    conv1=PReLU()(conv1)
    conv1=Conv3D(32,3,padding='same',kernel_initializer='he_normal')(conv1)
    conv1=BatchNormalization(axis=channel_axis)(conv1)
    conv1=PReLU()(conv1)
    conv1=Conv3D(32,3,padding='same',dilation_rate=(2, 2, 2),kernel_initializer='he_normal')( conv1)
    conv1=BatchNormalization(axis=channel_axis)(conv1)
    conv1=PReLU()(conv1)

    block1=dense_block(filters,conv1)
    conv1_1=BatchNormalization(axis=channel_axis)(block1)
    conv1_1=PReLU()(conv1_1)
    conv1_1=Conv3D(64,3,padding='same',kernel_initializer='he_normal')(conv1_1)
    conv1_1=BatchNormalization(axis=channel_axis)(conv1_1)
    conv1_1=PReLU()(conv1_1)
    conv1_1=Conv3D(64,3,padding='same',dilation_rate=(2, 2, 2), kernel_initializer='he_normal')(conv1_1)
    conv1_1=BatchNormalization(axis=channel_axis)(conv1_1)
    conv1_1=PReLU()(conv1_1)
    conv1_1=SpatialDropout3D(rate=0.2, data_format=data_format)(conv1_1)
    
    block2=dense_block(filters,conv1_1)
    conv2=concatenate([block2,block1],channel_axis)
    conv2=BatchNormalization(axis=channel_axis)(conv2)
    conv2=PReLU()(conv2)
    conv2=Conv3D(96, 3, padding='same',kernel_initializer='he_normal')(conv2)
    conv2=BatchNormalization(axis=channel_axis)(conv2)
    conv2=PReLU()(conv2)
    conv2=Conv3D(96, 3, padding='same',dilation_rate=(4, 4, 4),kernel_initializer='he_normal')(conv2)
    conv2=BatchNormalization(axis=channel_axis)(conv2)
    conv2=PReLU()(conv2)
    conv2=SpatialDropout3D(rate=0.2, data_format=data_format)(conv2)
    
    block3=dense_block(filters,conv2)
    conv3=concatenate([block3,block2],channel_axis)
    conv3=BatchNormalization(axis=channel_axis)(conv3)
    conv3=PReLU()(conv3)
    conv3=Conv3D(128, 3, padding='same',kernel_initializer='he_normal')(conv3)
    conv3=BatchNormalization(axis=channel_axis)(conv3)
    conv3=PReLU()(conv3)
    conv3=Conv3D(128, 3, padding='same',dilation_rate=(4, 4, 4),kernel_initializer='he_normal')(conv3)
    conv3=BatchNormalization(axis=channel_axis)(conv3)
    conv3=PReLU()(conv3)
    conv3=SpatialDropout3D(rate=0.2, data_format=data_format)(conv3)
    
    block4=dense_block(filters,conv3) #128+32
    conv4=concatenate([block4,block3],channel_axis)
    conv4=BatchNormalization(axis=channel_axis)(conv4)
    conv4=PReLU()(conv4)
    conv4=Conv3D(64, 1, padding='same',kernel_initializer='he_normal')(conv4)
    conv4=BatchNormalization(axis=channel_axis)(conv4)
    conv4=PReLU()(conv4)
    conv4=Conv3D(64, 3, padding='same',dilation_rate=(4, 4, 4),kernel_initializer='he_normal')(conv4)
    conv4=BatchNormalization(axis=channel_axis)(conv4)
    conv4=PReLU()(conv4)
    conv4=SpatialDropout3D(rate=0.2, data_format=data_format)(conv4)

    conv6=Conv3D(16,1,padding='same',dilation_rate=(4, 4, 4),kernel_initializer='he_normal')(conv1_1)
    conv7=Conv3D(16,1,padding='same',dilation_rate=(4, 4, 4),kernel_initializer='he_normal')(conv2)
    conv8=Conv3D(16,1,padding='same',dilation_rate=(4, 4, 4),kernel_initializer='he_normal')(conv3)
    conv9=Conv3D(16,1,padding='same',dilation_rate=(4, 4, 4),kernel_initializer='he_normal')(conv4)

    conv11=add([conv6,conv7,conv8,conv9])
    conv11=Conv3D(num_classes,1,padding='same',kernel_initializer='he_normal')(conv11)
    conv11=Activation('softmax')(conv11)
    model = Model(inputs = [inputs2], outputs = conv11)
    return model
