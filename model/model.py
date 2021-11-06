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

##additional sources ### 
# https://towardsdatascience.com/land-cover-classification-of-hyperspectral-imagery-using-deep-neural-networks-2e36d629a40e
# https://github.com/syamkakarla98/Hyperspectral_Image_Analysis_Simplified


### initial model for testing to be refined ## 
def SSH_CNN(n_classes=10, input_shape=(24,24,10,1)): 
    # inspiration from https://github.com/gokriznastic/HybridSN/blob/master/Hybrid-Spectral-Net.ipynb

    #encoder 
    input = Input(shape=input_shape, name="model_input")

    conv1 = Conv3D(8, kernel_size=(3,3,5), activation='relu', name='conv1') (input)
    conv2 = Conv3D(16, kernel_size=(3,3,1), activation='relu', name='conv2') (conv1)
    conv3 = Conv3D(32, kernel_size=(1,1,3), activation='relu', name='conv3') (conv2)
    conv3d_shape = conv3.shape
    conv3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv3)
    conv4 = Conv2D(64, kernel_size=(3,3), activation='relu', name='conv4') (conv3)

    flatten_layer = Flatten()(conv4)

    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu', name='dense1')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu', name='dense2')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=n_classes, activation='softmax', name='prediction')(dense_layer2)

    model = Model(input, output_layer)
    optimizer=tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return 


