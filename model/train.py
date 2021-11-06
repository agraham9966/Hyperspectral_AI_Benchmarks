import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as Kb
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.utils import plot_model

import numpy as np 
import random as random 
import os 

def batch_gen(X_names, y_names, batch_size=8):

    rows = 256 
    cols = 256 
    X_channels = 6
    y_channels = 6 #num output classes 

    while True: 

        c = list(zip(X_names, y_names))
        random.shuffle(c)
        _X_names, _y_names = zip(*c)

        X_ims = np.zeros((batch_size, rows, cols, X_channels))
        y_ims = np.zeros((batch_size, rows, cols, y_channels))

        ii=0

        for X_name, y_name in zip(_X_names,_y_names): 
            # print(X_name)
            # print(y_name)
            # print('end sample')
            X_ims[ii] = np.load(X_name)[:rows,:cols,:X_channels]
            y_ims[ii] = np.load(y_name)[:rows,:cols,:y_channels]
            #X_ims[ii] = img_to_array(X_name, dtype='float32').transpose(1,2,0)
            #y_ims[ii] = img_to_array(y_name, dtype='int32').transpose(1,2,0)

            ii+=1
            
            if ii>= batch_size: 
                yield(X_ims, y_ims)
                #reset for next batch 
                X_ims = np.zeros((batch_size, rows, cols, X_channels))
                y_ims = np.zeros((batch_size, rows, cols, y_channels))
                ii=0

    return 

def train_val_reader(X_train_path, y_train_path, X_val_path, y_val_path): 

    Xt_names = [os.path.join(X_train_path, i) for i in os.listdir(X_train_path)]
    yt_names = [os.path.join(y_train_path, i) for i in os.listdir(y_train_path)]

    Xv_names = [os.path.join(X_val_path, i) for i in os.listdir(X_val_path)]
    yv_names = [os.path.join(y_val_path, i) for i in os.listdir(y_val_path)]

    

    return Xt_names, yt_names, Xv_names, yv_names