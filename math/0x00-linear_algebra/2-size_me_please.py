#!/usr/bin/env python3

"""Calculates the shape of a matrix"""

def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape_matrix = []
    if (type(matrix) is list):
        shape_matrix.append(len(matrix))
        shape_matrix.extend(matrix_shape(matrix[0]))

    return shape_matrix
