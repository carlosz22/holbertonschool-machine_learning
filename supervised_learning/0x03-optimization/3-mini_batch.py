#!/usr/bin/env python3

"""Trains a loaded neural network model using mini-batch gradient descent"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch gradient descent
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        train_op = tf.get_collection('train_op')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        steps = X_train.shape[0] // batch_size

        if steps % batch_size != 0:
            steps = steps + 1
            end_extended = True
        else:
            end_extended = False

        for epoch in range(epochs + 1):

            cost_train, accuracy_train = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})

            cost_valid, accuracy_valid = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(accuracy_valid))

            if epoch < epochs:

                X_train_shuffled, Y_train_shuffled = shuffle_data(
                    X_train, Y_train)

                for step in range(steps):
                    start = step * batch_size

                    if step == steps - 1 and end_extended is True:
                        end = X_train.shape[0]
                    else:
                        end = step * batch_size + batch_size

                    X_minibatch = X_train_shuffled[start:end]
                    Y_minibatch = Y_train_shuffled[start:end]

                    sess.run(train_op,
                             feed_dict={x: X_minibatch, y: Y_minibatch})

                    if step != 0 and step + 1 % 100 == 0:
                        print("\tStep {}:".format(step + 1))

                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_minibatch, y: Y_minibatch})

                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
