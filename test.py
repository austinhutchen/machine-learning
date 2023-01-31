import tensorflow as tf
import tensorflow_datasets as tfds



##Goal of this program is to take in an arbitary dataset and decide on a function to map to output


print("TensorFlow version:", tf.__version__)
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

