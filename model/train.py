import tensorflow as tf 
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model, load_model
from model import SSH_CNN
import numpy as np 
import random as random 
import os 

X_train_path = r'C:\Users\James-007\Documents\Python_Scripts\Hyperspectral_AI_Benchmarks\datasets\Pavia\PaviaC_TrainTest\train\X'
y_train_path = r'C:\Users\James-007\Documents\Python_Scripts\Hyperspectral_AI_Benchmarks\datasets\Pavia\PaviaC_TrainTest\train\y'

def batch_gen(X_names, y_names, batch_size=8):

    rows = 25
    cols = 25 
    X_channels = 102
    y_channels = 9 #num output classes 

    while True: 

        c = list(zip(X_names, y_names))
        random.shuffle(c)
        _X_names, _y_names = zip(*c)

        X_ims = np.zeros((batch_size, rows, cols, X_channels, 1))
        y_ims = np.zeros((batch_size, y_channels))

        ii=0

        for X_name, y_name in zip(_X_names,_y_names): 
            X_ims[ii] = np.load(X_name)[:rows,:cols,:X_channels,:]
            y_ims[ii] = np.load(y_name)[:y_channels]

            ii+=1
            
            if ii>= batch_size: 
                yield(X_ims, y_ims)
                #reset for next batch 
                X_ims = np.zeros((batch_size, rows, cols, X_channels, 1))
                y_ims = np.zeros((batch_size, y_channels))
                ii=0

    return 

def train_val_reader(X_train_path, y_train_path, X_val_path, y_val_path): 

    Xt_names = [os.path.join(X_train_path, i) for i in os.listdir(X_train_path)]
    yt_names = [os.path.join(y_train_path, i) for i in os.listdir(y_train_path)]

    Xv_names = [os.path.join(X_val_path, i) for i in os.listdir(X_val_path)]
    yv_names = [os.path.join(y_val_path, i) for i in os.listdir(y_val_path)]

    return Xt_names, yt_names, Xv_names, yv_names


def train(): 
    # to update when gpu is installed 
    model = SSH_CNN(n_classes=9, input_shape=(25,25,102,1))
    
    Xt_names, yt_names, Xv_names, yv_names = train_val_reader(X_train_path, y_train_path, X_train_path, y_train_path)
    
    
    model.fit(batch_gen(Xt_names, yt_names, batch_size=8),
                                        steps_per_epoch = int(float(len(Xt_names))/float(50)),
                                        validation_data=batch_gen(Xv_names, yv_names, batch_size=50),
                                        validation_steps = int(float(len(Xv_names))/float(50)),
                                        epochs = 50)

    return 