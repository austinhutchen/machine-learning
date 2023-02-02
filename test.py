import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.naive_bayes import BernoulliNB

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


def get_likelihood(features, label_indices, smoothing=0):
    #  Compute likelihood based on training samples
    #  @param features: matrix of features
    # @param label_indices: grouped sample indices by class
    # @param smoothing: integer, additive smoothing parameter
    # @return: dictionary, with class as key, corresponding
    #   conditional probability P(feature|class) vector
    #    as value
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / \
            (total_count + 2 * smoothing)
    return likelihood


if __name__ == "__main__":
    # training set x ( think of as rows/m in mxn matrix)
    x_train = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0]
    ])
    # training set y (think of as columns or variabls in mxn matrix.)
    y_train = ['T', 'N', 'Y', 'Y']
    # test set
    x_test = np.array([[1, 1, 0]])
    clf = BernoulliNB(alpha=1.0, fit_prior=True)
    clf.fit(x_train, y_train)
    prediction = clf.predict_proba(x_test)
    labels_indices = get_label_indices(y_train)
    print('Label Indices: ', labels_indices, '\n')
    print("Prediction:", prediction, '\n')
