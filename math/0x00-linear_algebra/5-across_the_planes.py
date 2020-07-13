#!/usr/bin/env python3

"""Add two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Add two matrices elemnt-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = []
    for i in range(len(mat1)):
        result.append([])
        for j in range(len(mat1[0])):
            result[i].append(mat1[i][j] + mat2[i][j])
    return result
