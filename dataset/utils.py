import tensorflow as tf

# -------------- utils --------------

def create_dataset(args):
    return tf.keras.utils.image_dataset_from_directory(
        directory=args.dataset,
        labels=None,
        label_mode=None,
        color_mode="rgba",
        batch_size=args.batch,
        image_size=(args.image_size,)*2,
        shuffle=True
    )