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
import cv2
import argparse

from loader import make_generator_model, make_discriminator_model, load_dataset, GAN
from loader import discriminator_loss, generator_loss

if __name__ == "__main__":
    Arg = argparse.ArgumentParser()
    Arg.add_argument('--ckptdir', default="./ckpt/checkpoint", help="the dir you save the weight")
    Arg.add_argument('--datasetdir', default='./CastleDataset/', help="Your dataset dir")
    Arg.add_argument('--outputdir', default='./', help="save Your infered img in dir")
    
    args = Arg.parse_args()
    checkpoint_filepath = args.ckptdir
    img_output_path = args.outputdir

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    GAN_model = GAN(generator=generator, discriminator=discriminator)
    GAN_model.compile(
        d_optimizer=tf.keras.optimizers.Adam(1e-4),
        g_optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    random_input = tf.random.normal(shape=(8, 100))
    img = GAN_model.generator(random_input)
    output = GAN_model.discriminator(img)
    GAN_model.load_weights(checkpoint_filepath).expect_partial()

    img = GAN_model.generator(random_input)

    img = img.numpy()
    if(img is not None):
        print(img.shape)
        print("GET img")
    for i in range(len(img)):
        cv2.imwrite("{:04d}.png".format(i),img[i])
    '''
    '''