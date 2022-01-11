import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.ops.array_ops import rank_internal
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.training.tracking.base import NoRestoreSaveable

# load dataset
def load_dataset(dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = dir,
    label_mode=None,
    batch_size = 16,
    image_size = (512,512),
    shuffle = True
    )
    return train_ds

# g model
def make_generator_model():
    model = keras.Sequential(
        [
            layers.Dense(10*10*256, use_bias=False, input_shape=(100,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((10, 10, 256)),

            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(3, (2, 2), strides=(1, 1), padding='valid', use_bias=False,activation="relu"),
        ]
    )
    
    return model

# d model
def make_discriminator_model():
    model = tf.keras.Sequential(
        [
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[512, 512, 3]),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(1),
        ]
    )
    return model


# returns a helper function to compute cross entropy loss
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

''' NOT READY TO USE!!!
# call back used in tf.model.fit()
class end_of_epoch_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
'''


class GAN(keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.counter = 0
    
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, d_loss=discriminator_loss, g_loss=generator_loss):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        random_input = tf.random.normal(shape=(batch_size, 100))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generate_img = self.generator(random_input)

            real_output = self.discriminator(data)
            fake_output = self.discriminator(generate_img)

            gen_loss = self.g_loss(fake_output)
            disc_loss = self.d_loss(real_output, fake_output)
        
        gradient_of_gen = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        gradient_of_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)

        self.g_optimizer.apply_gradients(zip(gradient_of_gen, self.generator.trainable_weights))
        self.d_optimizer.apply_gradients(zip(gradient_of_disc, self.discriminator.trainable_weights))

        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return{
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }