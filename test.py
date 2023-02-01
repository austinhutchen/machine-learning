import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


# Goal of this program is to take in an arbitary dataset and decide on a function to map to output


print("TensorFlow version:", tf.__version__)
print("Num CPUs Available: ", len(
    tf.config.experimental.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

def get_label_indices(labels):
#    """
    # Group samples based on their labels and return indices ...     @param labels: list of labels
   # @return: dict, {class1: [indices], class2: [indices]} ...     """
 from collections import defaultdict
 label_indices = defaultdict(list)
 for index, label in enumerate(labels):
      label_indices[label].append(index)
 return label_indices

if __name__ == "__main__":
 # training set x ( think of as rows/m in mxn matrix)
 x_train = np.array([[0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 1, 0]])
 # training set y (think of as columns or variabls in mxn matrix.)
 y_train = ['T', 'N', 'Y', 'Y']
 # test set
 x_test = np.array([1, 1, 0])