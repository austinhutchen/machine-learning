import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__)

print("Num CPUs Available: ", len(
    tf.config.experimental.list_physical_devices('CPU')))


