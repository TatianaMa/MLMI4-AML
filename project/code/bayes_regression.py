import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import variational as vl

LEARNING_RATE = 1e-5
def create_model(features, labels, params):
    """
    Builds the computation graph for the baseline model.

    :param features: input data to the network

    :param params: dictionary of parameters passed to model_fn

    """

    prior_fns = {
        "gaussian": vl.create_gaussian_prior,
        "mixture": vl.create_mixture_prior
    }

    try:
        prior_fn = prior_fns[params["prior"]]

        print("Using {} prior!".format(params["prior"]))
    except KeyError as e:

        print("No prior specified (possibilities: {}). Using standard Gaussian!".format(prior_fns.keys()))
        prior_fn = None

    input_shape = [-1] + params["input_dims"]

        # Input layer
    input_layer = tf.reshape(features, input_shape)

    logits_list = []

    klds_list = []

    for i in range(params["num_mc_samples"]):

        dense1, kld1 = vl.variational_dense(
            inputs=input_layer,
            name="variational_dense_1",
            units=params["hidden_units"],
            prior_fn=prior_fn,
            params=params
        )

        # dense2, kld2 = vl.variational_dense(
        #     inputs=dense1,
        #     name="variational_dense_2",
        #     units=params["hidden_units"],
        #     prior_fn=prior_fn,
        #     params=params
        # )


        # Output Layer
        logits, kld3 = vl.variational_dense(inputs=dense1,
                                            units=1,
                                            activation=None,
                                            name="variational_dense_out",
                                            prior_fn=prior_fn,
                                            params=params
        )

        logits_list.append(logits)
        klds_list.append(sum([kld1, kld3]))

    kld = sum(klds_list) #/ float(params["num_mc_samples"])

    return logits_list, kld


def bayes_regression_model_fn(features, labels, mode, params):

    if "learning_rate" not in params:
        raise KeyError("No learning rate specified!")

    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]),
        "momentum": tf.train.MomentumOptimizer(learning_rate=params["learning_rate"], momentum=0.9, use_nesterov=True),
        "adam": tf.train.AdamOptimizer(learning_rate=params["learning_rate"]),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])
    }

    pred_list, kld = create_model(features, labels, params)

    predictions = sum(pred_list)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # the global step is the batch number
    batch_number = tf.train.get_global_step() // params["kl_coeff_decay_rate"]

    kl_coeffs = {
        # if batch_number > 30, we just set the coefficient to 0, otherwise to (1/2)^batch_number
        "geometric": tf.pow(0.5, tf.cast(batch_number + 1, tf.float32)) * tf.cast(tf.less(batch_number, 30), tf.float32),

        # Since we rely on the user passing the number of batches as a param, we need to check
        "uniform": 1./float(params["num_batches"]) if "kl_coeff" in params and "num_batches" in params else 1.
    }

    try:
        kl_coeff = kl_coeffs[params["kl_coeff"]]

    except KeyError as e:

        raise KeyError("kl_coeff must be one of {}".format(kl_coeffs.keys()))

    loss, kl, loglik = vl.ELBO_with_MSE(predictions_list=pred_list,
                                        kl_divergences=[kld],
                                        kl_coeff=kl_coeff,
                                        labels=tf.reshape(labels, [-1, 1]))


    if mode == tf.estimator.ModeKeys.TRAIN:
        try:
            optimizer = optimizers[params["optimizer"]]
        except KeyError as e:
            raise KeyError("No optimizer specified! Possibilities: {}".format(optimizers.keys()))

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        # Summary statistic for TensorBoard
        tf.summary.scalar('train_KL_coeff', kl_coeff)
        tf.summary.scalar('train_KL', kl)
        tf.summary.scalar('train_log_likelihood', loglik)


        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:

        layers = ["variational_dense_1",  "variational_dense_out"]

        samples = []

        for layer in layers:
            for w in ["weight_", "bias_"]:
                var = []
                for theta in ["mu", "rho"]:
                    var.append([v for v in tf.trainable_variables() if v.name == layer + "/" + w + theta + ":0"][0])

                mu = var[0]
                sigma = tf.nn.softplus(var[1])

                sample = tfd.Normal(loc=mu, scale=sigma).sample()

                samples.append(tf.reshape(sample, [-1]))

        eval_hooks = []

        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            summary_op=tf.summary.histogram("weight/hist", tf.concat(samples, axis=0)))

        eval_hooks.append(eval_summary_hook)

        eval_metric_ops = {
            "ELBO": loss
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=eval_hooks)
