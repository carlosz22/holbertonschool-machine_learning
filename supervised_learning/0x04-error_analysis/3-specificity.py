#!/usr/bin/env python3

"""Calculates the specificity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
      - confusion is a confusion numpy.ndarray of shape (classes, classes)}
       where row indices represent the correct labels and column indices
        represent the predicted labels
      - classes is the number of classes
      Returns: a numpy.ndarray of shape (classes,) containing the
       specificity of each class
    """
    real = np.sum(confusion, axis=1)
    pred = np.sum(confusion, axis=0)

    total = np.sum(confusion)
    true_positive = np.diag(confusion)

    false_positive = pred - true_positive
    actual_negative = total - real
    true_negative = actual_negative - false_positive

    return true_negative / actual_negative
