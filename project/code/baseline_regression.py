# ==============================================================================
#
# This implementation is based on the official TensorFlow implementation that
# can be found at https://github.com/tensorflow/models/tree/master/official/mnist
#
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

LEARNING_RATE = 1e-3

def create_model(features, params):
    """
    Builds the computation graph for the baseline model.

    :param features: input data to the network

    :param params: dictionary of parameters passed to model_fn

    """

    input_shape = [-1, 1]

    # Input layer
    input_layer = tf.reshape(features, input_shape)

    # Dense Layer #1
    dense1 = Dense(
        units=params['hidden_units'],
        activation=tf.nn.relu
    ).apply(input_layer)

    #dense1 = Dropout(0.3).apply(dense1)

    # Dense Layer #2
    dense2 = Dense(
        units=params['hidden_units'],
        activation=tf.nn.relu
    ).apply(dense1)

    #dense2 = Dropout(0.3).apply(dense2)

    # Output Layer
    logits = Dense(
        units=2
    ).apply(dense2)

    return logits



def baseline_regression_model_fn(features, labels, mode, params):
    """
    This function will handle the training, evaluation and prediction procedures.
    """
    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]),
        "momentum": tf.train.MomentumOptimizer(learning_rate=params["learning_rate"], momentum=0.9, use_nesterov=True),
        "adam": tf.train.AdamOptimizer(learning_rate=params["learning_rate"]),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])
    }

    logits = create_model(features, params)


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

    logits = tf.reshape(logits, [-1, 2])

    mus = logits[:, 0]
    sigmas = tf.nn.softplus(logits[:, 1])

    # (proportional) Log probability of diagonal Gaussian
    neg_log_prob = tf.log(sigmas) + tf.square(mus - tf.reshape(labels, [-1])) / tf.square(sigmas)

    loss = tf.reduce_sum(neg_log_prob)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        #mae = tf.metrics.mean_absolute_error(labels=labels,
        #                                     predictions=logits)

        # Summary statistic for TensorBoard
        #tf.summary.scalar('train_mae', mae)

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "mae": tf.keras.metrics.mean_absolute_error(labels=labels,
                                                        predictions=logits),
            "mse": loss
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
