import tensorflow as tf

def gen_opt():
    return tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def disc_opt():
    return tf.keras.optimizers.Adam(2e-4, beta_1=0.5)