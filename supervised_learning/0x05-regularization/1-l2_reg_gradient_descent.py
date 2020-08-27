#!/usr/bin/env python3

"""Updates the weights and biases of a neural network
 using gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
     using gradient descent with L2 regularization
        - Y is a one-hot numpy.ndarray of shape (classes, m)
         that contains the correct labels for the data
            - classes is the number of classes
            - m is the number of data points
        - weights is a dictionary of the weights and biases
         of the neural network
        - cache is a dictionary of the outputs of each layer
         of the neural network
        - alpha is the learning rate
        - lambtha is the L2 regularization parameter
        - L is the number of layers of the network
        - The neural network uses tanh activations on each layer
         except the last, which uses a softmax activation
        - The weights and biases of the network should be updated
         in place
    """
    m = Y.shape[1]
    for i in reversed(range(L)):
        lay = str(i + 1)
        lay_min_1 = str(i)
        if i == (L - 1):
            dZ = cache['A' + str(L)] - Y
        else:
            dZ = dA * (np.tanh(cache['A' + lay]) ** 2)
        dW = (np.matmul(dZ, cache['A' + lay_min_1].T) / m) + \
             ((lambtha / m) * weights['W' + lay])
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(weights['W' + lay].T, dZ)
        weights['W' + lay] = weights['W' + lay] - \
            (alpha * dW)
        weights['b' + lay] = weights['b' + lay] - \
            (alpha * db)
