#!/usr/bin/env python3

"""Multiplies two matrices"""


def mat_mul(mat1, mat2):
    """Multiplies two matrices"""
    if (len(mat1[0]) != len(mat2)):
        return None

    new_matrix = []
    for i in range(len(mat1)):
        inner_list = []
        for j in range(len(mat2[0])):
            inner_result = 0
            for k in range(len(mat1[0])):
                inner_result += mat1[i][k] * mat2[k][j]
            inner_list.append(inner_result)
        new_matrix.append(inner_list)
    return new_matrix
