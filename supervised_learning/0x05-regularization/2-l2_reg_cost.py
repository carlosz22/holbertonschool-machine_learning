#!/usr/bin/env python3

"""Calculates the cost of a neural network with L2 regularization
 using Tensorflow"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization
     using Tensorflow
        - cost is a tensor containing the cost of the network
         without L2 regularization
        Returns: a tensor containing the cost of the network
         accounting for L2 regularization
    """
    total_loss = cost + tf.losses.get_regularization_loss(scope=None)
    return total_cost
