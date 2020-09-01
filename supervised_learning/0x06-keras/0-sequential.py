#!/usr/bin/env python3

"""Builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
        - nx is the number of input features to the network
        - layers is a list containing the number of nodes in each
         layer of the network
        - activations is a list containing the activation functions
         used for each layer of the network
        - lambtha is the L2 regularization parameter
        - keep_prob is the probability that a node will be kept for
         dropout
        - You are not allowed to use the Input class
        Returns: the keras model
    """

    reg_l2 = K.regularizers.l2(lambtha)

    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i],
                      activation=activations[i],
                      kernel_regularizer=reg_l2,
                      input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(layers[i],
                      activation=activations[i],
                      kernel_regularizer=reg_l2))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(keep_prob))
    return model
