import numpy as np
from parso import parse
import tensorflow as tf
import cv2

import argparse

from dataset import create_dataset, augmentation
from model import Pix2Pix
from model import gen_opt, disc_opt
# ------------------- arg -------------------
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="input", help="dataset 的路徑")
parser.add_argument('--batch', type=int, default=1, help="batch size")
parser.add_argument('--epoch', type=int, default=2, help="number of epoch")
parser.add_argument('--g-step', type=int, default=3, help="the step generator train every batch")
parser.add_argument('--d-step', type=int, default=3, help="the step discriminator train every batch")
parser.add_argument('--checkpoint', type=int, default=1000, help="the interval of saving checkpoint")
parser.add_argument('--checkpoint-img', type=int, default=1000, help="the interval of print out image")
parser.add_argument('--image-size', type=int, default=512,
                    help="圖片要resize的大小(一個整數) eg. 1024=1024x1024 ")

args = parser.parse_args()

# =================== main ===================
dataset = create_dataset(args)
dataset = dataset.map(augmentation)

model = Pix2Pix(args)
model.compile(G_opter=tf.keras.optimizers.Adam(2e-4, beta_1=0.5), 
              D_opter=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))
model.fit(dataset, epochs=args.epoch)