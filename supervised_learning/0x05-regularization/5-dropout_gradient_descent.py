#!/usr/bin/env python3

"""Conducts forward propagation using Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
     Updates the weights of a neural network
      with Dropout regularization using gradient descent
        - Y is a one-hot numpy.ndarray of shape (classes, m)
         that contains the correct labels for the data
            - classes is the number of classes
            - m is the number of data points
        - weights is a dictionary of the weights and biases
         of the neural network
        - cache is a dictionary of the outputs and dropout
         masks of each layer of the neural network
        - alpha is the learning rate
        - keep_prob is the probability that a node will be kept
        - L is the number of layers of the network
        - All layers use the tanh activation function except
         the last, which uses the softmax activation function
        - The weights of the network should be updated in place
    """
    m = Y.shape[1]
    for i in reversed(range(L)):
        lay = str(i + 1)
        lay_min_1 = str(i)
        if i == (L - 1):
            dZ = cache['A' + str(L)] - Y
        else:
            dZ = dA * (1 - (cache['A' + lay] ** 2))
            dZ *= cache['D' + lay]
            dZ /= keep_prob
        dW = (np.matmul(dZ, cache['A' + lay_min_1].T)) / m
        db = (np.sum(dZ, axis=1, keepdims=True)) / m
        dA = np.matmul(weights['W' + lay].T, dZ)
        weights['W' + lay] = weights['W' + lay] - \
            (alpha * dW)
        weights['b' + lay] = weights['b' + lay] - \
            (alpha * db)
