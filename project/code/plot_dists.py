import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

def plot_dists(bayes, baseline, dropout):
    binwidth = 0.001
    bins = np.arange(-1.25, 1.25 + binwidth, binwidth)
    # bins = np.arange(min(samples_val), max(samples_val) + binwidth, binwidth)

    # Bayes
    tf.reset_default_graph()
    checkpoint = tf.train.get_checkpoint_state(bayes)
    print(checkpoint.model_checkpoint_path)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        layers = ["variational_dense_1", "variational_dense_2", "variational_dense_out"]

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
        samples_val = sess.run(tf.concat(samples, axis=0))
        plt.hist(samples_val, alpha=0.5, bins=bins, range=binwidth, label='BBB')

    # Dropout
    tf.reset_default_graph()
    checkpoint = tf.train.get_checkpoint_state(dropout)
    print(checkpoint.model_checkpoint_path)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)
        graph = tf.get_default_graph()
        # for t_var in tf.trainable_variables():
        #     print(t_var)
        layers = ["dense", "dense_1", "dense_2"]
        weights = []
        for layer in layers:
            for w in ["kernel:0", "bias:0"]:
                name = layer + "/" + w
                # print(name)
                var = [v for v in tf.trainable_variables() if v.name == name][0]
                weights.append(tf.reshape(var, [-1]))
        weights_val = sess.run(tf.concat(weights, axis=0))
        plt.hist(weights_val, alpha=0.5, bins=bins, range=binwidth, label='Dropout')

    # Baseline
    tf.reset_default_graph()
    checkpoint = tf.train.get_checkpoint_state(baseline)
    print(checkpoint.model_checkpoint_path)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)
        graph = tf.get_default_graph()
        # for t_var in tf.trainable_variables():
        #     print(t_var)
        layers = ["dense", "dense_1", "dense_2"]
        weights = []
        for layer in layers:
            for w in ["kernel:0", "bias:0"]:
                name = layer + "/" + w
                # print(name)
                var = [v for v in tf.trainable_variables() if v.name == name][0]
                weights.append(tf.reshape(var, [-1]))
        weights_val = sess.run(tf.concat(weights, axis=0))
        plt.hist(weights_val, alpha=0.5, bins=bins, range=binwidth, label='FFNN')

    plt.legend(loc='upper right')
    plt.xlabel('Weight')
    plt.ylabel('Density')
    plt.savefig("figs/weight_dist_600.png")
    # plt.show()

plot_dists("models/bayes_mnist_800_600", "models/baseline_800_600", "models/dropout_mnist_800_600")
