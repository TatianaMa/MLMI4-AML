import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

LEARNING_RATE = 1e-5
def create_model(features, params, labels):
    """
    Builds the computation graph for the baseline model.

    :param features: input data to the network

    :param params: dictionary of parameters passed to model_fn

    """

    prior_fns = {
        "gaussian": create_gaussian_prior,
        "mixture": create_mixture_prior
    }

    try:
        prior_fn = prior_fns[params["prior"]]

        print("Using {} prior!".format(params["prior"]))
    except KeyError as e:

        print("No prior specified (possibilities: {}). Using standard Gaussian!".format(prior_fns.keys()))
        prior_fn = None

    input_shape = [-1, 28 * 28]

    # Input layer
    input_layer = tf.reshape(features, input_shape)

    dense1, kld1 = variational_dense(
        inputs=input_layer,
        units=params["hidden_units"],
        name="variational_dense_1",
        prior_fn=create_mixture_prior,
        params=params
    )

    dense2, kld2 = variational_dense(
        inputs=dense1,
        units=params["hidden_units"],
        name="variational_dense_2",
        prior_fn=create_mixture_prior,
        params=params
    )

    # Output Layer
    logits, kld3 = variational_dense(inputs=dense2,
                                     units=10,
                                     activation=None,
                                     name="variational_dense_out",
                                     prior_fn=create_mixture_prior,
                                     params=params
    )


    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    kl_divergence = kld1 + kld2 + kld3

    # the global step is the batch number
    batch_number = tf.train.get_global_step()

    kl_coeffs = {
        # if batch_number > 30, we just set the coefficient to 0, otherwise to (1/2)^batch_number
        "geometric": tf.pow(0.5, tf.cast(batch_number + 1, tf.float32)) * tf.squeeze(tf.cast(tf.greater(batch_number, 30), tf.float32)),

        # Since we rely on the user passing the number of batches as a param, we need to check
        "uniform": 1./float(params["num_batches"]) if "kl_coeff" in params and "num_batches" in params else 1.
    }

    try:
        kl_coeff = kl_coeffs[params["kl_coeff"]]

    except KeyError as e:

        raise KeyError("kl_coeff must be one of {}".format(kl_coeffs.keys()))

    loss = kl_coeff * kl_divergence + loss

    return logits, loss

def create_weights_and_biases(units_prev, units_next):
    # ========================
    # Weights
    # ========================

    mu_init = tf.initializers.random_normal(mean=0., stddev=.1)
    rho_init = tf.initializers.random_normal(mean=-3., stddev=.1)

    weight_mu = tf.get_variable(name="weight_mu", shape=[units_prev, units_next], initializer=mu_init)
    weight_rho = tf.get_variable(name="weight_rho", shape=[units_prev, units_next], initializer=rho_init)

    # sigma = log(1 + exp(rho))
    weight_sigma = tf.nn.softplus(weight_rho)

    # w = mu + sigma * epsilon
    weight_dist = tfd.Normal(loc=weight_mu, scale=weight_sigma)

    # ========================
    # Biases
    # ========================

    bias_mu = tf.get_variable(name="bias_mu", shape=[units_next], initializer=mu_init)
    bias_rho = tf.get_variable(name="bias_rho", shape=[units_next], initializer=rho_init)

    # sigma = log(1 + exp(rho))
    bias_sigma = tf.nn.softplus(bias_rho)

    # b = mu + sigma * epsilon
    bias_dist = tfd.Normal(loc=bias_mu, scale=bias_sigma)

    return weight_dist, bias_dist

def variational_dense(inputs,
                  units,
                  name="variational_dense",
                  activation=tf.nn.relu,
                  prior_fn=None,
                  params=None):
    """
    prior_fn(units_prev, units_next) -> tfd.Distribution
    """
    with tf.variable_scope(name):
        weight_dist, bias_dist = create_weights_and_biases(
            units_prev=inputs.shape[1],
            units_next=units
        )

        weights = weight_dist.sample()
        biases = bias_dist.sample()

        dense = tf.matmul(inputs, weights) + biases

        if activation is not None:
            dense = activation(dense)

        if prior_fn is None:
            prior = create_gaussian_prior({"mu":0., "sigma":0.})
        else:
            prior = prior_fn(params)

        weight_prior_lp = prior.log_prob(weights)
        bias_prior_lp = prior.log_prob(biases)

        weight_var_post_lp = weight_dist.log_prob(weights)
        bias_var_post_lp = bias_dist.log_prob(biases)

        kl_divergence = tf.reduce_sum(weight_var_post_lp - weight_prior_lp)
        kl_divergence += tf.reduce_sum(bias_var_post_lp - bias_prior_lp)

    return dense, kl_divergence

def create_gaussian_prior(params):
    prior = tfd.Normal(loc=params["mu"], scale=tf.exp(-params["sigma"]))
    return prior

def create_mixture_prior(params):
    prior = tfd.Mixture(
        cat=tfd.Categorical(probs=[params["mix_prop"], 1. - params["mix_prop"]]),
        components=[
            tfd.Normal(loc=0., scale=tf.exp(-params["sigma1"])),
            tfd.Normal(loc=0., scale=tf.exp(-params["sigma2"])),
        ])
    return prior


def bayes_mnist_model_fn(features, labels, mode, params):

    if "learning_rate" not in params:
        raise KeyError("No learning rate specified!")

    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]),
        "adam": tf.train.AdamOptimizer(learning_rate=params["learning_rate"]),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])
    }

    logits, loss = create_model(features, params, labels)
    predictions = tf.argmax(input=logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        try:
            optimizer = optimizers[params["optimizer"]]
        except KeyError as e:
            raise KeyError("No optimizer specified! Possibilities: {}".format(optimizers.keys()))

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predictions)

        # Summary statistic for TensorBoard
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions)
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
