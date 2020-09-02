#!/usr/bin/env python3

"""Save the best iteration of the model"""
import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                save_best=False,
                filepath=None,
                verbose=True,
                shuffle=False):
    """
    Save the best iteration of the model
        - save_best is a boolean indicating whether to save
         the model after each epoch if it is the best
        - a model is considered the best if its validation loss
         is the lowest that the model has obtained
        - filepath is the file path where the model should be saved
    """

    callbacks = []

    def learning_rate_decay(epoch):
        """Calculates the inverse time decay in each epoch"""
        return alpha / (1 + decay_rate * epoch)

    if filepath:
        mcp_save = K.callbacks.ModelCheckpoint(filepath,
                                               save_best_only=save_best)
        callbacks.append(mcp_save)

    if validation_data and learning_rate_decay:
        lr_decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                     verbose=1)
        callbacks.append(lr_decay)

    if validation_data and early_stopping:
        es = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                       mode='min')
        callbacks.append(es)

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callbacks)

    return history
