from gc import callbacks
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import glob
import matplotlib.pyplot as plt
import os
import PIL
import time

from IPython import display
from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator

from loader import make_generator_model, make_discriminator_model, load_dataset, GAN
from loader import discriminator_loss, generator_loss
# from loader import end_of_epoch_callback

# load dataset from directory
DATASET_PATH = "./CastleDataset/"
checkpoint_filepath = "./ckpt/checkpoint"
train_dataset = load_dataset(DATASET_PATH)
# for data in train_dataset.take(1).as_numpy_iterator():
  # print(data.shape)

# G 和 D 模型
generator = make_generator_model()
discriminator = make_discriminator_model()

GAN_model = GAN(generator=generator, discriminator=discriminator)
GAN_model.compile(
    d_optimizer=tf.keras.optimizers.Adam(1e-4),
    g_optimizer=tf.keras.optimizers.Adam(1e-4)
)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose=0,
    save_freq=30,
)
# my_callback = end_of_epoch_callback()

history = GAN_model.fit(train_dataset, epochs=1000,callbacks=[ckpt_callback])
print(history.history)
