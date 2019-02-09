import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def create_model(features, params):
    """
    Builds the computation graph for the baseline model.

    :param features: input data to the network

    :param params: dictionary of parameters passed to model_fn

    """

    input_shape = [-1, 28 * 28]

    # Input layer
    input_layer = tf.reshape(features["x"], input_shape)

    weights, biases = create_weights_and_biases(inputs=input_layer,
                                                units=params["hidden_units"])

    # Output Layer
    logits = tf.layers.dense(inputs=dense2, units=10)

    return logits

def create_weights_and_biases(inputs, units):
    # ========================
    # Weights
    # ========================

    weight_mu = tf.Variable("mu_matrix", [units, inputs], initializer=tf.zero_initializer)
    weight_rho = tf.Variable("sigma_matrix", [units, inputs], initializer=tf.zero_initializer)

    # sigma = log(1 + exp(rho))
    weight_sigma = log1pexp(weight_rho)

    # w = mu + sigma * epsilon
    weight_sample = # TODO

    # ========================
    # Biases
    # ========================

    bias_mu = tf.Variable("mu_matrix", [units], initializer=tf.zero_initializer)
    bias_rho = tf.Variable("sigma_matrix", [units], initializer=tf.zero_initializer)

    # sigma = log(1 + exp(rho))
    bias_sigma = log1pexp(bias_rho)

    # epsilon ~ N(0, I)
    bias_epsilon = tfd.Normal(loc=0, scale=[1]).sample([units])

    # w = mu + sigma * epsilon
    bias_sample = bias_mu + tf.math.multiply(bias_sigma, bias_epsilon)

@tf.custom_gradient
def log1pexp(x):
    """
    Calculates the function log(1 + exp(x)), providing an efficient, numerically stable
    gradient function for it as well.

    Taken from https://www.tensorflow.org/api_docs/python/tf/custom_gradient
    """
    e = tf.exp(x)

    def grad(dy):
        return dy * (1 - 1 / (1 + e))

    return tf.log(1 + e), grad

@tf.custom_gradient
def location_scale_reparametrisation(*x):
    """
    Calculates the function mu + sigma * epsilon, providing also the Bayesian gradient.
    """

    mu, sigma = x

    # epsilon ~ N(0, I)
    epsilon = tfd.Normal(loc=0, scale=[1]).sample(tf.shape(mu))

    w = mu + tf.math.multiply(sigma, epsilon)

    def grad(*dy):
        return None # TODO

    return w, grad


def bayes_mnist_model_fn(features, labels, mode, params):
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
