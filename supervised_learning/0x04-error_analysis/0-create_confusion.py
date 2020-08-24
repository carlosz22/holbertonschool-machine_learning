#!/usr/bin/env python3

"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
        - labels is a one-hot numpy.ndarray of shape (m, classes)
         containing the correct labels for each data point
        - m is the number of data points
        - classes is the number of classes
        - logits is a one-hot numpy.ndarray of shape (m, classes)
         containing the predicted labels
    Returns: a confusion numpy.ndarray of shape (classes, classes)
     with row indices representing the correct labels
     and column indices representing the predicted labels
    """
    num_classes = labels.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes))
    pred_classes = np.argmax(logits, axis=1)
    true_classes = np.argmax(labels, axis=1)

    for i in range(len(pred_classes)):
        confusion_matrix[true_classes[i], pred_classes[i]] += 1

    return confusion_matrix
