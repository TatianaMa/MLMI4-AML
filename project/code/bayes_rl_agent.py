import tensorflow as tf

import variational as vl

NUM_ACTIONS = 2

def create_model(features, params):

    prior_fns = {
        "gaussian": vl.create_gaussian_prior,
        "mixture": vl.create_mixture_prior
    }

    try:
        prior_fn = prior_fns[params["prior"]]

        if "verbose" in params and params["verbose"]:
            print("Using {} prior!".format(params["prior"]))
    except KeyError as e:

        print("No prior specified (possibilities: {}). Using standard Gaussian!".format(prior_fns.keys()))
        prior_fn = None


    input_layer = tf.cast(tf.reshape(features, [-1, params["context_size"] + NUM_ACTIONS]), tf.float32)

    expected_rewards = []
    klds_list = []

    for i in range(params["num_mc_samples"]):

        # Dense Layer #1
        dense1, kl1 = vl.variational_dense(
            inputs=input_layer,
            name="variational_dense_1",
            units=params['hidden_units'],
            activation=tf.nn.relu
        )

        # Dense Layer #2
        dense2, kl2 = vl.variational_dense(
            inputs=dense1,
            name="variational_dense_2",
            units=params['hidden_units'],
            activation=tf.nn.relu
        )


        # Output Layer
        expected_reward, kl3 = vl.variational_dense(
            inputs=dense2,
            name="variational_dense_out",
            units=1,
            activation=None
        )

        expected_rewards.append(expected_reward)
        klds_list.append(sum([kl1, kl2, kl3]))

    kld = sum(klds_list)

    return expected_rewards, kld


def bayes_rl_agent_model_fn(features, labels, mode, params):

    if "learning_rate" not in params:
        raise KeyError("No learning rate specified!")

    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]),
        "momentum": tf.train.MomentumOptimizer(learning_rate=params["learning_rate"], momentum=0.9, use_nesterov=True),
        "adam": tf.train.AdamOptimizer(learning_rate=params["learning_rate"]),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])
    }

    expected_rewards, kld = create_model(features, params)

    expected_reward = sum(expected_rewards) / float(params["num_mc_samples"])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=expected_reward)

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

    loss, kl, loglik = vl.ELBO_with_MSE(predictions_list=expected_rewards,
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

        eval_metric_ops = {
            "MSE": loss
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
