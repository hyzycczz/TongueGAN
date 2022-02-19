from ast import Load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import yaml
from yaml import Loader

from loader import load_dataset, load_discriminator, load_generator, WGAN, GANMonitor, generator_loss, discriminator_loss

if __name__ == "__main__":
    ''' --------------DCGAN START HERE--------------
    # load configuration from "config.yml"
    with open("config.yml", 'r') as f:
        config = yaml.load(f, Loader=Loader)

    dataset = load_dataset(config['dataset'])

    print(dataset.element_spec)

    latent_dim = 128
    Generator = load_generator(latent_dim)
    Discriminator = load_discriminator()

    epochs = 100  # In practice, use ~100 epochs

    gan = GAN(discriminator=Discriminator, generator=Generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    gan.fit(
        dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
    )

    # --------------DCGAN END-------------- '''


    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=Loader)

    dataset = load_dataset(config['dataset'])
    print(dataset.element_spec)

    latent_dim = 128
    Generator = load_generator(latent_dim)
    Discriminator = load_discriminator()

    epochs = 2000

    gan = WGAN(discriminator=Discriminator, 
        generator=Generator,
        latent_dim=latent_dim,
        discriminator_extra_steps=5)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    gan.fit(
        dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim)]
    )