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
parser.add_argument('--image-size', type=int, default=512,
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

model = Pix2Pix(args)
model.compile(G_opter=tf.keras.optimizers.Adam(2e-4, beta_1=0.5), 
              D_opter=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))


log_dir = os.path.join(args.log_dir, args.model_name)
ckpt_dir = os.path.join(args.ckpt_dir, args.model_name)

writer = tf.summary.create_file_writer(os.path.join(log_dir,datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
checkpoint = tf.train.Checkpoint(G_opter=model.G_opter,
                                 D_opter=model.D_opter,
                                 G_model=model.G,
                                 D_model=model.D)

for step, images in dataset.repeat().take(args.step).enumerate(start=1):
    mask, image = images

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_img = model.G(mask, training=True)
            
        real_pred = model.D(mask, image, training=True)
        fake_pred = model.D(mask, fake_img, training=True)
        total_gen_loss,  gan_loss, l1_loss = model.G_loss(fake_pred, fake_img, image)
        disc_loss = model.D_loss(real_pred, fake_pred)

    G_gradients = g_tape.gradient(total_gen_loss, model.G.trainable_variables)
    D_gradients = d_tape.gradient(disc_loss, model.D.trainable_variables)
    model.G_opter.apply_gradients(zip(G_gradients, model.G.trainable_variables))
    model.D_opter.apply_gradients(zip(D_gradients, model.D.trainable_variables))

    if (step % args.log_interval == 0):
        with writer.as_default():
            tf.summary.scalar("total_gen_loss", total_gen_loss, step)
            tf.summary.scalar("gan_loss", gan_loss, step)
            tf.summary.scalar("l1_loss", l1_loss, step)
            tf.summary.scalar("disc_loss", disc_loss, step)

    if (step % args.log_img_interval == 0):
        with writer.as_default():
            tf.summary.image("mask", mask[:1], step)
            tf.summary.image("real_image", image[:1], step)
            tf.summary.image("fake_image", fake_img[:1], step)
    
    if (step % args.ckpt_interval == 0):
        checkpoint.save(file_prefix=ckpt_dir)
    
    # Training Animation 
    

'''
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir,
                                                   save_freq=args.ckpt_interval)
model.fit(dataset, epochs=args.epoch, callbacks=[ckpt_callback])
'''