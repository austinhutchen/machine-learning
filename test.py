import tensorflow as tf
import tensorflow_datasets as tfds


if __name__ == '_main':
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    tf.config.list_physical_devices('GPU')
