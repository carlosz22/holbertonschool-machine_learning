#!/usr/bin/env python3

"""save_model and load_model functions"""
import tensorflow.keras as K


def save_model(network, filename):
    """Save an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model"""
    model = K.models.load_model(filename)
    return model
