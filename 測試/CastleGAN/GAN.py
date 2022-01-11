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
import argparse

from IPython import display
from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator

from loader import make_generator_model, make_discriminator_model, load_dataset, GAN
from loader import discriminator_loss, generator_loss
# from loader import end_of_epoch_callback

if __name__ == "__main__":
    Arg = argparse.ArgumentParser()
    Arg.add_argument('--ckptdir', default="./ckpt/checkpoint", help="the dir you save the weight")
    Arg.add_argument('--datasetdir', default='./CastleDataset/', help="Your dataset dir")
    Arg.add_argument('--saveFreq',type=int, default=30, help="the number of frequence you saving data")
    Arg.add_argument('--epoch',type=int, default=1000, help="the number of epoch")
    
    args = Arg.parse_args()
    # load dataset from directory
    DATASET_PATH = args.datasetdir
    checkpoint_filepath = args.ckptdir
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
        save_freq=args.saveFreq,
    )
    # my_callback = end_of_epoch_callback()

    history = GAN_model.fit(train_dataset, epochs=args.epoch,callbacks=[ckpt_callback])
    print(history.history)
