import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display


class CRNN(tf.keras.layers.Layer):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=16 , kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)  
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=32 , kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=64 , kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv2d_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bn_4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv2d_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bn_5 = tf.keras.layers.BatchNormalization(axis=-1)

        self.LSTM_1 = tf.keras.layers.LSTM(units=1024,return_sequences=True)
        self.LSTM_2 = tf.keras.layers.LSTM(units=1024,return_sequences=True)
        

        self.deconv2d_5 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bnt_5 = tf.keras.layers.BatchNormalization(axis=-1)
        self.deconv2d_4 = tf.keras.layers.Conv2DTranspose(filters=64 , kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bnt_4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.deconv2d_3 = tf.keras.layers.Conv2DTranspose(filters=32 , kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bnt_3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.deconv2d_2 = tf.keras.layers.Conv2DTranspose(filters=16 , kernel_size=(2,3), strides=(1,2),output_padding=(0,1),activation="elu")
        self.bnt_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.deconv2d_1 = tf.keras.layers.Conv2DTranspose(filters=1  , kernel_size=(2,3), strides=(1,2),activation="elu")
        self.bnt_1 = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        # Encode
        e1 = self.bn_1(self.conv2d_1(inputs))
        e2 = self.bn_2(self.conv2d_2(e1))
        e3 = self.bn_3(self.conv2d_2(e2))
        e4 = self.bn_4(self.conv2d_2(e3))
        e5 = self.bn_5(self.conv2d_2(e4))

        # reshape
        # x5 = tf.transpose(x5,perm=[0,2,1,3])
        reshape_e5 = tf.reshape(e5,shape=[e5.shape[0],-1,1024])

        #LSTM
        lstm1 = self.LSTM_1(reshape_e5)
        lstm2 = self.LSTM_2(lstm1)

        # reshape
        reshape_lstm2 = tf.reshape(lstm2,shape=[lstm2.shape[0],-1,4,256])

        # Decode
        d5 = tf.concat([reshape_lstm2,e5],axis=-1)
        d5 = self.bnt_5(self.deconv2d_5(d5))
        d4 = tf.concat([d5,e4],axis=-1)
        d4 = self.bnt_4(self.deconv2d_5(d4))
        d3 = tf.concat([d4,e3],axis=-1)
        d3 = self.bnt_3(self.deconv2d_5(d3))
        d2 = tf.concat([d3,e2],axis=-1)
        d2 = self.bnt_2(self.deconv2d_5(d2))
        d1 = tf.concat([d2,e1],axis=-1)
        d1 = self.bnt_1(self.deconv2d_5(d1))

        #reshape
        output = tf.reshape(d1,shape=[d1.shape[0],d1.shape[1],d1.shape[2]])
        return output