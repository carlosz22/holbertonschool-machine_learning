#!/usr/bin/env python3

"""Transpose a matrix"""


def matrix_transpose(matrix):
    """Transpose a matrix"""
    transpose = []
    for j in range(len(matrix[0])):
        transpose.append([])
        for i in range(len(matrix)):
            transpose[j].append(matrix[i][j])
    return transpose
