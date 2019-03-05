from bayes_mnist import create_model

def prune_weights(features, labels, params):
    logits, loss = create_model(features, params, labels)

    # Weights summary
    layers = ["variational_dense_1", "variational_dense_2", "variational_dense_out"]

    weights = []
    for layer in layers:
        for w in ["weight_", "bias_"]:
            var = []
            for theta in ["mu", "rho"]:
                var.append([v for v in tf.trainable_variables() if v.name == layer + "/" + w + theta + ":0"][0])

            mu = var[0]
            rho = var[1]

            weights.append(mu)
            weights.append(rho)

            # sigma = tf.nn.softplus(var[1])

            # sample = tfd.Normal(loc=mu, scale=sigma).sample()
            #
            # samples.append(tf.reshape(sample, [-1]))
    print(len(weights))
    # tf.summary.histogram("weight/hist", tf.concat(samples, axis=0))
    # train_hooks = []

    # train_summary_hook = tf.train.SummarySaverHook(
    #     save_steps=1,
    #     summary_op=tf.summary.histogram("weight/hist", tf.concat(samples, axis=0)))

    # train_hooks.append(train_summary_hook)
