#!/usr/bin/env python3

"""save_config and load_config functions"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model’s configuration in JSON format"""
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    return None


def load_config(filename):
    """Loads a model with a specific configuration"""
    with open(filename, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = K.models.model_from_json(loaded_model_json)
    return model
