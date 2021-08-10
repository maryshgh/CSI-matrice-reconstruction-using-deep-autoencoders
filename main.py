# Import related libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
tf.compat.v1.reset_default_graph()

# Parameter initialization
envir_type = 'indoor' #'indoor' for indoor 5.3GHz picocellular, 'outdoor' for 300MHz outdoor
# image parameters
image_height = 32
image_width = 32
image_channels = 2 
image_size = image_height*image_width*image_channels
# network parameters
residual_num = 2  # Number of residual units
encoded_dim = 512  #the number of parameters in encoded CSI Matrice, compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

# Clone dataset - CSI matrice built based on COST2100 model for MIMO scenarios
!git clone https://github.com/sydney222/Python_CsiNet.git

# Install google drive in colab
from google.colab import drive
drive.mount('/content/drive')

# Bulid the autoencoder model of CsiNet
# Define residual unit
def residual_block(x):
  shortcut = x
  x = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
        
  x = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
        
  x = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
  x = BatchNormalization()(x)

  x = add([shortcut, x])
  x = LeakyReLU()(x)
  return x

def residual_network(x, residual_num, encoded_dim):
    # Encoder convolutional layer followed with batch normalization and leaky ReLU as activation 
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Flattening the output of Convolutional layer
    x = Reshape((image_size,))(x)
    # change the dimension of the output of encoder to the desired dimension defined as encoded_dim
    encoded = Dense(encoded_dim, activation='linear')(x) # encoder output
    
    # Decoder part
    x = Dense(image_size, activation='linear')(encoded)
    x = Reshape((image_channels, image_height, image_width,))(x)
    # Add residula_num number of residula blocks
    for i in range(residual_num):
        x = residual_block(x)
    # Convert the dimension of the encoder output to the dimension of the CSI matrice
    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x

# Define network input
network_input = Input(shape=(image_channels, image_height, image_width))
# Define network output
network_output = residual_network(network_input, residual_num, encoded_dim)
autoencoder = Model(inputs=[network_input], outputs=[network_output])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
if envir_type == 'indoor':
    mat = sio.loadmat('drive/MyDrive/data/DATA_Htrainin.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('drive/MyDrive/data/DATA_Hvalin.mat')
    x_validation = mat['HT'] # array
    mat = sio.loadmat('drive/MyDrive/data/DATA_Htestin.mat')
    x_test = mat['HT'] # array

elif envir_type == 'outdoor':
    mat = sio.loadmat('drive/MyDrive/data/DATA_Htrainout.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('drive/MyDrive/data/DATA_Hvalout.mat')
    x_validation = mat['HT'] # array
    mat = sio.loadmat('drive/MyDrive/data/DATA_Htestout.mat')
    x_test = mat['HT'] # array
    
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), image_channels, image_height, image_width))  # adapt this if using `channels_first` image data format
x_validation = np.reshape(x_validation, (len(x_validation), image_channels, image_height, image_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), image_channels, image_height, image_width))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=200,
                shuffle=True,
                validation_data=(x_validation, x_validation))
