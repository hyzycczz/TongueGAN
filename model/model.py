import tensorflow as tf
from tensorflow import keras

import os


class G_model(keras.Model):
    '''
    pix2pix Generator 512x512
    '''

    def __init__(self):
        super().__init__()
        self.OUTPUT_CHANNELS = 3
        self.initializer = tf.random_normal_initializer(0., 0.02)

        self.Encoder = [
            self.downsample(64, 4, apply_batchnorm=False),  # 256x256
            self.downsample(128, 4),                        # 128x128
            self.downsample(256, 4),                        # 64 x64
            self.downsample(512, 4),                        # 32 x32
            self.downsample(1024, 4),                       # 16 x16
            self.downsample(1024, 4),                       # 8  x8
            self.downsample(1024, 4),                       # 4  x4
            self.downsample(1024, 4),                       # 2  x2
            self.downsample(1024, 4)                        # 1  x1
        ]

        self.Decoder = [
            self.upsample(1024, 4, apply_dropout=True),     # 2  x2
            self.upsample(1024, 4, apply_dropout=True),     # 4  x4
            self.upsample(1024, 4, apply_dropout=True),     # 8  x8
            self.upsample(1024, 4),                         # 16 x16
            self.upsample(512, 4),                          # 32 x32
            self.upsample(256, 4),                          # 64 x64
            self.upsample(128, 4),                          # 128x128
            self.upsample(64, 4),                           # 256x256
        ]

        # Back to 512x512x3
        self.last_output = tf.keras.layers.Conv2DTranspose(filters=self.OUTPUT_CHANNELS,
                                                           kernel_size=4,
                                                           strides=2,
                                                           padding='same',
                                                           kernel_initializer=self.initializer,
                                                           activation='tanh')

    def call(self, mask, training=False):
        x = mask
        skips = []  # skip connection

        # Encoder
        for encode in self.Encoder:
            x = encode(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Decoder
        for decode, skip in zip(self.Decoder, skips):
            x = decode(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = self.last_output(x)

        return x

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = self.initializer

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = self.initializer

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result


class D_model(keras.Model):
    '''
    pix2pix Discriminator
    '''

    def __init__(self):
        super().__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)

        self.down_layers = [
            self.downsample(64, 4, apply_batchnorm=False),  # 256x256
            self.downsample(128, 4),                        # 128x128
            self.downsample(256, 4),                        # 64 x64
            self.downsample(512, 4),                        # 32 x32
            tf.keras.layers.ZeroPadding2D(),                # 34 x34
            tf.keras.layers.Conv2D(512, 4, strides=1,
                                   kernel_initializer=self.initializer,
                                   use_bias=False),          # 31 x31
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.ZeroPadding2D(),                # 33x33
            tf.keras.layers.Conv2D(1, 4, strides=1,
                                   kernel_initializer=self.initializer)
        ]

        return

    def call(self, mask, image, training=False):
        x = tf.concat( (mask,image), axis=-1)
        for layer in self.down_layers:
            x = layer(x)

        return x

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = self.initializer

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result


class Pix2Pix(keras.Model):
    def __init__(self, args):
        super(Pix2Pix, self).__init__()
        self.G = G_model()
        self.D = D_model()

        return

    def call(self, images, training=False):
        mask, image = images
        output = self.G(mask)
        return output

    def compile(self, G_opter, D_opter):
        super(Pix2Pix, self).compile()
        self.G_opter = G_opter
        self.D_opter = D_opter
        self.G_loss = self.generator_loss
        self.D_loss = self.discriminator_loss
        return

    def generator_loss(self, fake_pred, fake_image, real_image):
        LAMBDA = 100
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        gan_loss = loss_object(tf.ones_like(
            fake_pred), fake_pred)
        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(real_image - fake_image))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss