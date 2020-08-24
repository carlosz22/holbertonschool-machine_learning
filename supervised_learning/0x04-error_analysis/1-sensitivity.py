#!/usr/bin/env python3

"""Calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
        - confusion is a confusion numpy.ndarray of shape (classes, classes)
         where row indices represent the correct labels and column indices
          represent the predicted labels
        - classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,) containing
         the sensitivity of each class
    """
    column_matrix = np.sum(confusion, axis=1)
    diagonal = np.diag(confusion)
    return diagonal / column_matrix
