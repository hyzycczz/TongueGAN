from pydoc import Helper
import tensorflow as tf
import random
import cv2


@tf.function
def augmentation(img):

    mask = img[:, :, :, 3:]/255
    img = img[:, :, :, :3]/255
    if (tf.random.uniform(()) > 0.5):
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    output = tf.image.random_brightness(img, 0.9)
    output = tf.image.random_saturation(output, 0.6, 1.5)
    output = tf.image.random_contrast(output, 0.8, 2)

    return mask, tf.math.multiply(output, mask)
