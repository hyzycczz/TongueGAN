from ast import Num
import numpy as np
import tensorflow as tf
from tensorflow import keras

class PG_G(keras.Model):
    '''
    PG_GAN Generator
    '''
    def __init__(self, latent_dim, layers):
        super(PG_G, self).__init__()
        self.latent_dim = latent_dim
        self.layers = layers
        self.resolution = 2 ** (layers+1)

        self.upsample = keras.layers.UpSampling2D(size=(2,2),interpolation="nearest")

        self.Blks = [self.create_block(block) for block in range(layers)] #依據layer數建立 block
        self.Blks_rgb = [self.create_toRGB(block) for block in range(layers)] #依據layer數建立 rgb 層
    
    def call(self, x, curr_layer, alpha):
        for i in range(curr_layer):
            x = self.Blks[i](x)
            x = self.upsample(x)
        
        _x = self.Blks_rgb[i](x)
        x = self.Blks[curr_layer](x)
        x = self.Blks_rgb[i](x)

        x = x * alpha + _x * (1-alpha)
        return x
    ''' --------------- 生成器 utils ---------------'''
    def create_block(self,curr_layer):
        filter = min(16 * 2 **(self.layers-curr_layer-1), self.resolution//2)
        if(curr_layer == 0):
            blk = keras.Sequential([
                keras.layers.Dense(4 * 4 * 512, input_shape=(self.latent_dim,)),
                keras.layers.LeakyReLU(),
                keras.layers.Reshape((4, 4, 512)),
                keras.layers.Conv2D(filter, (3, 3), strides=1, padding='same'),
                keras.layers.LeakyReLU(),
            ])
        else:
            blk = keras.Sequential([
                keras.layers.Conv2D(filter, (3, 3), strides=1, padding='same'),
                keras.layers.LeakyReLU(),
                keras.layers.Conv2D(filter, (3, 3), strides=1, padding='same'),
                keras.layers.LeakyReLU(),
            ])
            
        return blk

    def create_toRGB(current_layer):
        return keras.layers.Conv2D(3, 1, 1, activation="linear")


class PG_D(keras.Model):
    '''
    PG_GAN Discriminator
    '''
    def __init__(self, layers):
        super(PG_D, self).__init__()
        self.layers = layers
        self.resolution = 2 ** (layers+1)

        self.Blks = [self.create_block(block) for block in range(layers)]
        self.Blks_rgb = [self.create_fromRGB(block) for block in range(layers)]
        self.pool = keras.layers.AvgPool2D((4, 4), strides=(2, 2), padding="same")

    def call(self, x, curr_layer, alpha):
        


    def create_block(self, curr_layer):
        filter = min(16 * 2**curr_layer, self.resolution//2)
        if(curr_layer == self.layers - 1):
            blk = keras.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(1)
            ])
        else:
            blk = keras.Sequential([
                keras.layers.Conv2D(filter, 3, strides=1, padding='same'),
                keras.layers.LeakyReLU(),
                keras.layers.Dropout(0.3),
                keras.layers.Conv2D(min(self.resolution,filter*2), 3, strides=1, padding='same'),
                keras.layers.LeakyReLU(),
                keras.layers.Dropout(0.3),
            ])
        return blk

    def create_fromRGB(self, current_layer):
        filter = min(16 * 2**current_layer, self.resolution//2)
        fromRBG = keras.Sequential([
            keras.layers.Conv2D(filter, 1, 1, activation="linear"),
            keras.layers.LeakyReLU(),
        ])
        return fromRBG

    
class PG_WGAN(keras.Model):
    def __init__(
        self,
        latent_dim,
        layer,
        fade_step,
        GP_weight = 10,
        d_steps = 3
    ):
        super(PG_WGAN, self).__init__()
        # general GAN
        self.generator = PG_G(latent_dim, layer)
        self.discriminator = PG_D(layer)
        self.latent_dim = latent_dim
        # PG-GAN
        self.layer = layer
        self.current_layer = 0
        self.fade_step = fade_step
        self.fade_count = 0
        # WGAN
        self.GP_weight = GP_weight
        self.d_steps = d_steps


    def compile(self, g_opt, d_opt, g_loss, d_loss):
        super(PG_WGAN, self).compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def gradient_penalty(self, batch_size, real_img, fake_img):
        test = 1
        return

    def train_d(fake, real):


        return
    
    def train_g():
        return
    
    def train_step(self, real_img):
        if not isinstance(real_img, tuple):
            raise Exception("Input type is wrong!!")

        batchsize = tf.shape(real_img)[0]
        
        # train discriminator first
        for i in range(self.d_steps):
            fake_img = self.generator(tf.random.normal(shape=(batchsize, self.latent_dim)))
            self.train_discriminator(fake_img, real_img)

        self.train_g()