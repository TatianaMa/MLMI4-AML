# Adapted from https://github.com/tensorflow/tensorflow/issues/16646#issuecomment-384935839
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_weights():
    binwidth = 0.1

    # colors = sns.color_palette("Blues")
    colors = sns.cubehelix_palette(8, start=.5, rot=-.75)

    snrs = get_snrs("models/bayes_mnist_800_100")
    sns.distplot(snrs, hist = False, label="100", color=colors[7])

    snrs = get_snrs("models/bayes_mnist_800_200")
    sns.distplot(snrs, hist = False, label="200", color=colors[6])

    snrs = get_snrs("models/bayes_mnist_800_300")
    sns.distplot(snrs, hist = False, label="300", color=colors[5])

    snrs = get_snrs("models/bayes_mnist_800_400")
    sns.distplot(snrs, hist = False, label="400", color=colors[4])

    snrs = get_snrs("models/bayes_mnist_800_500")
    sns.distplot(snrs, hist = False, label="500", color=colors[3])

    snrs = get_snrs("models/bayes_mnist_800_600")
    sns.distplot(snrs, hist = False, label="600", color=colors[2])

    # fig = dist_plot.get_figure()
    plt.xlabel('Signal-To-Noise Ratio (dB)')
    plt.ylabel('Density')
    plt.legend(title = '# Epochs')
    plt.savefig("distplot.png")

    # plt.title('Histogram of the signal-to-noise ratio over all weights')
    # plt.savefig("prune_weights.png")
    # plt.show()

def get_snrs(model_dir):
    tf.reset_default_graph()
    checkpoint = tf.train.get_checkpoint_state(model_dir)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        layers = ["variational_dense_1", "variational_dense_2", "variational_dense_out"]

        # Obtain SNRs and calculate pruning threshold
        snrs = [] # Signal to noise ratios
        for layer in layers:
            for w in ["weight_", "bias_"]:
                var = []
                for theta in ["mu", "rho"]:
                    var.append([v for v in tf.trainable_variables() if v.name == layer + "/" + w + theta + ":0"][0])
                mu = var[0]
                sigma = tf.nn.softplus(var[1])
                snr = tf.math.scalar_mul(10., tf.math.log(tf.math.divide(tf.math.abs(mu), sigma)))
                snrs.append(tf.reshape(snr, [-1]))
        snrs = tf.concat(snrs, axis=0)
        snrs_val = sess.run(snrs)
        return snrs_val

plot_weights()
