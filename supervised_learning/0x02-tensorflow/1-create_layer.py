#!/usr/bin/env python3

"""Creates a layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer
        - prev is the tensor output of the previous layer
        - n is the number of nodes in the layer to create
        - activation is the activation function that the layer should use
        - used tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
         to implement He et. al initialization for the layer weights
        Returns: the tensor output of the layer
    """

    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=weights_initializer,
                            name="layer")

    return layer(prev)
