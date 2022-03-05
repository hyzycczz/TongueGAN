import numpy as np
import tensorflow as tf
import os
import datetime
import argparse

from dataset import create_dataset, augmentation
from model import Pix2Pix

# ------------------- arg -------------------
parser = argparse.ArgumentParser()

parser.add_argument('--model-name', type=str, default="pix2pix", help="name of model(use in saving weight respectively)")
parser.add_argument('--dataset', type=str, default="input", help="dataset 的路徑")
parser.add_argument('--image-size', type=int, default=16,
                    help="圖片要resize的大小(一個整數) eg. 512=512x512 ")
parser.add_argument('--batch', type=int, default=1, help="batch size")
parser.add_argument('--step', type=int, default=2, help="number of model interate")
parser.add_argument('--log-dir', type=str, default="./log", help="the dirction of log")
parser.add_argument('--log-interval', type=int, default=10, help="the interval of logging loss")
parser.add_argument('--log-img-interval', type=int, default=500, help="the interval of logging image")

parser.add_argument('--ckpt-dir', type=str, default="./checkpoint", help="the direction of saving checkpoint")
parser.add_argument('--ckpt-interval', type=int, default=500, help="the interval of saving checkpoint")

args = parser.parse_args()

# =================== main ===================
dataset = create_dataset(args)
dataset = dataset.map(augmentation)

for data in dataset:
    mask, d = data
    print(d)

    break