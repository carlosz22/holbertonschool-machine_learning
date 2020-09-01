#!/usr/bin/env python3

"""Builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
        - nx is the number of input features to the network
        - layers is a list containing the number of nodes
         in each layer of the network
        - activations is a list containing the activation functions
         used for each layer of the network
        - lambtha is the L2 regularization parameter
        - keep_prob is the probability that a node will be kept
         for dropout
        - You are not allowed to use the Sequential class
        Returns: the keras model
    """

    reg_l2 = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))

    for i in range(len(layers)):
        if i == 0:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=reg_l2)(inputs)
        elif i == len(layers) - 1:
            outputs = K.layers.Dense(layers[i], activation=activations[i])(x)
            break
        else:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=reg_l2)(x)
        x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model
