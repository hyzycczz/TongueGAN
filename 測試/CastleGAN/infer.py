import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import yaml
from yaml import Loader
import cv2
import matplotlib.pyplot as plt
import numpy as np

from loader import load_dataset

loss = keras.losses.BinaryCrossentropy()

true = np.array([[1]],dtype=np.float64)
pred = np.linspace(0, 1, 100,dtype=np.float64)
L = []

for i in pred:
    p = [[i]]
    L.append(loss(true,p))

plt.plot(pred, L)
plt.show()