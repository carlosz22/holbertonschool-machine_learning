#!/usr/bin/env python3

"""Defines a Neuron for binary classification"""
import numpy as np


class Neuron:
    """Class Neuron"""

    def __init__(self, nx):
        """Constructor method"""

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(*(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W getter"""
        return self.__W

    @property
    def b(self):
        """b getter"""
        return self.__b

    @property
    def A(self):
        """A getter"""
        return self.__A

    def forward_prop(self, X):
        """Forward propagation"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost"""
        m = Y.shape[1]
        cost = -(np.sum(np.multiply(
            Y, np.log(A)) + np.multiply((1 - Y), np.log(1.0000001 - A)))) / m
        return cost

    def evaluate(self, X, Y):
        """Performs binary classification
        Outputs prediction matrix and cost"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        y_hat = self.__A.round().astype(int)
        return y_hat, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Performs gradient descent"""
        m = Y.shape[1]
        dZ = A - Y
        db = np.sum(dZ) / m
        dW = np.matmul(X, dZ.T) / m
        self.__b = self.__b - alpha * db
        self.__W = self.__W - alpha * dW.T
