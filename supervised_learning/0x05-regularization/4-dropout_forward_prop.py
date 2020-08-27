#!/usr/bin/env python3

"""Conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
        - X is a numpy.ndarray of shape (nx, m) containing the
         input data for the network
            - nx is the number of input features
            - m is the number of data points
        - weights is a dictionary of the weights and biases
         of the neural network
        - L the number of layers in the network
        - keep_prob is the probability that a node will be kept
        - All layers except the last should use the tanh activation function
        - The last layer should use the softmax activation function
        Returns: a dictionary containing the outputs of each layer and
         the dropout mask used on each layer (see example for format)
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        lay = str(i + 1)
        lay_min_1 = str(i)
        Zl = np.matmul(weights['W' + lay],
                       cache['A' + lay_min_1]) + weights['b' + lay]
        if i == L - 1:
            t = np.exp(Zl)
            cache['A' + lay] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache['A' + lay] = np.tanh(Zl)
            cache['D' + lay] = (np.random.rand(cache['A' + lay].shape[0],
                                cache['A' + lay].shape[1]) < keep_prob) * 1
            cache['A' + lay] = np.multiply(cache['A' + lay], cache['D' + lay])
            cache['A' + lay] /= keep_prob
    return cache
