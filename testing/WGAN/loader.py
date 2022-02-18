import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# load dataset
def load_dataset(config):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory = config["dir"],
        label_mode=config["label_mode"],
        batch_size = config['batch'],
        image_size = config['img_size'],
        shuffle = config['suffle']
    )
    return train_ds

