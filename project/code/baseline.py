# ==============================================================================
#
# This implementation is based on the official TensorFlow implementation that
# can be found at https://github.com/tensorflow/models/tree/master/official/mnist
#
# ==============================================================================

import numpy as np
import tensorflow as tf

LEARNING_RATE = 1e-4

def create_model(features, params):
    """
    Builds the computation graph for the baseline model.

    :param features: input data to the network

    :param params: dictionary of parameters passed to model_fn

    """

    input_shape = [-1, 28 * 28]

    # Input layer
    input_layer = tf.reshape(features["x"], input_shape)

    # Dense Layer #1
    dense1 = tf.layers.dense(
        inputs=input_layer,
        units=params['hidden_units'],
        activation=tf.nn.relu
    )

    # Dense Layer #2
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=params['hidden_units'],
        activation=tf.nn.relu
    )

    # Output Layer
    logits = tf.layers.dense(inputs=dense2, units=10)

    return logits



def baseline_model_fn(features, labels, mode, params):
    """
    This function will handle the training, evaluation and prediction procedures.
    """

    logits = create_model(features, params)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predictions["classes"])

        # Summary statistic for TensorBoard
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
