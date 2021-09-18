import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display


class CRNN(keras.layers.Layer):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,3), strides=(1,2))
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,3), strides=(1,2))
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,3), strides=(1,2))
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.conv2d_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,3), strides=(1,2))
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.conv2d_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,3), strides=(1,2))
        self.bn_5 = tf.keras.layers.BatchNormalization()

        self.LSTM_1 = tf.keras.layers.LSTM(units=1024)
        self.LSTM_2 = tf.keras.layers.LSTM(units=1024)
        

        deconv2d_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2,3), strides=(1,2))
        self.bnt_5 = tf.keras.layers.BatchNormalization()
        deconv2d_2 = tf.keras.layers.Conv2DTranspose(filters=64 , kernel_size=(2,3), strides=(1,2))
        self.bnt_5 = tf.keras.layers.BatchNormalization()
        deconv2d_3 = tf.keras.layers.Conv2DTranspose(filters=32 , kernel_size=(2,3), strides=(1,2))
        self.bnt_5 = tf.keras.layers.BatchNormalization()
        deconv2d_4 = tf.keras.layers.Conv2DTranspose(filters=16 , kernel_size=(2,3), strides=(1,2))
        self.bnt_5 = tf.keras.layers.BatchNormalization()
        deconv2d_5 = tf.keras.layers.Conv2DTranspose(filters=1  , kernel_size=(2,3), strides=(1,2))

    def call(self, inputs):
        # CNN down
        x1 = self.bn_1(self.conv2d_1(inputs))
        x2 = self.bn_2(self.conv2d_2(x1))
        x3 = self.bn_3(self.conv2d_2(x2))
        x4 = self.bn_4(self.conv2d_2(x3))
        x5 = self.bn_5(self.conv2d_2(x4))

        # reshape
        x5 = tf.transpose(x5,perm=[0,2,1,3])
        x5 = tf.reshape(x5,shape=[x5.shape[0],-1,1024])

        #LSTM
        lstm1 = self.LSTM_1(x5)
        lstm2 = self.LSTM_2(lstm1)

        # reshape
        