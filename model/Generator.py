import numpy as np
import tensorflow as tf
from tensorflow import keras


class PG_G(keras.Model):
    def __init__(self) -> None:
        super().__init__()