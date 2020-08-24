#!/usr/bin/env python3

"""Calculates the weighted moving average"""


def moving_average(data, beta):
    """Calculates the weighted moving average
        - data is the list of data to calculate the moving average of
        - beta is the weight used for the moving average
        Your moving average calculation should use bias correction
        Returns: a list containing the moving averages of data
    """
    moving_avg = []
    value = 0

    for i in range(len(data)):
        value = beta * value + (1 - beta) * data[i]
        value_corrected = value / (1 - (beta ** (i + 1)))
        moving_avg.append(value_corrected)
    return moving_avg
