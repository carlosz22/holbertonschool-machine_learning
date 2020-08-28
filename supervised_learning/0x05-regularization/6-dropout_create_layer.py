#!/usr/bin/env python3

"""Creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout
        - prev is a tensor containing the output of the previous layer
        - n is the number of nodes the new layer should contain
        - activation is the activation function that should be used
         on the layer
        - keep_prob is the probability that a node will be kept
        Returns: the output of the new layer
    """
    regul = tf.contrib.layers.l2_regularizer(lambtha)

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    model = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regul)

    dropout = tf.layers.Dropout(keep_prob)

    return dropout(model(prev))
