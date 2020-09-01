#!/usr/bin/env python3

"""Train the model using early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Train the model using early stopping
        - early_stopping is a boolean that indicates
         whether early stopping should be used
        - early stopping should only be performed if
         validation_data exists
        - early stopping should be based on validation loss
        - patience is the patience used for early stopping
    """

    callbacks = []

    if validation_data and early_stopping:
        es = K.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       mode='min')
        callbacks.append(es)

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    return history
