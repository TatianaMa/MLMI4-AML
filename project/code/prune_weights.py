# Adapted from https://github.com/tensorflow/tensorflow/issues/16646#issuecomment-384935839
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def prune_weights(model, model_dir, pruning_percentile, plot_hist=False):
    if model == "bayes_mnist":
        return prune_weights_bayes(model_dir, pruning_percentile, plot_hist)
    else:
        return prune_weights_other(model_dir, pruning_percentile, plot_hist)

def prune_weights_bayes(model_dir, pruning_percentile, plot_hist):
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
        pruning_threshold = sess.run(tf.contrib.distributions.percentile(snrs, q=pruning_percentile, interpolation='lower'))
        print("PRUNINNG THRESHOLD")
        print(pruning_threshold)

        if plot_hist:
            binwidth = 1
            snrs_val = sess.run(snrs)
            # plt.hist(snrs_val, bins=np.arange(min(snrs_val), max(snrs_val) + binwidth, binwidth))
            sns.distplot(snrs_val, hist=False, color=sns.cubehelix_palette(8, start=.5, rot=-.75)[7])
            plt.axvline(x=pruning_threshold, color='tab:red')
            plt.xlabel('Signal-To-Noise Ratio (dB)')
            plt.ylabel('Density')
            # plt.title('Histogram of the signal-to-noise ratio over all weights')
            fig = plt.gcf()
            # fig.set_size_inches(50,35)
            # plt.savefig("prune_weights.eps")
            plt.show()

        # Prune weights using the obtained threshold
        for layer in layers:
            for w in ["weight_", "bias_"]:
                var = []
                for theta in ["mu", "rho"]:
                    var.append([v for v in tf.trainable_variables() if v.name == layer + "/" + w + theta + ":0"][0])
                mu = var[0]
                sigma = tf.nn.softplus(var[1])
                snr = tf.math.scalar_mul(10., tf.math.log(tf.math.divide(tf.math.abs(mu), sigma)))
                mask = tf.dtypes.cast(tf.math.less(snr, pruning_threshold), dtype=tf.float32)
                mu_updated = tf.math.multiply(mu, mask)
                var_updated = var[0].assign(mu_updated)
                sess.run(var_updated)
                rho_updated = tf.contrib.distributions.softplus_inverse(tf.math.multiply(sigma, mask))
                var_updated = var[1].assign(rho_updated)
                sess.run(var_updated)

        # Save pruned model in a new directory
        new_dir = model_dir + '_pruned_' + str(pruning_percentile)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        saver.save(sess, new_dir + '/model.ckpt')
        print("FINISHED PRUNING")
        return new_dir

def prune_weights_other(model_dir, pruning_percentile, plot_hist):
    tf.reset_default_graph()
    checkpoint = tf.train.get_checkpoint_state(model_dir)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        layers = ["dense", "dense_1", "dense_2"]
        weights = []
        for layer in layers:
            for w in ["kernel:0", "bias:0"]:
                name = layer + "/" + w
                var = tf.math.abs([v for v in tf.trainable_variables() if v.name == name][0])
                weights.append(tf.math.abs(tf.reshape(var, [-1])))
        weights = tf.concat(weights, axis=0)
        pruning_threshold = sess.run(tf.contrib.distributions.percentile(weights, q=pruning_percentile, interpolation='lower'))

        for layer in layers:
            for w in ["kernel:0", "bias:0"]:
                name = layer + "/" + w
                var = [v for v in tf.trainable_variables() if v.name == name][0]
                var_abs = tf.math.abs(var)
                mask = tf.dtypes.cast(tf.math.greater(var_abs, pruning_threshold), dtype=tf.float32)
                var_new = tf.math.multiply(var, mask)
                var_update = var.assign(var_new)
                sess.run(var_update)

        # Save pruned model in a new directory
        new_dir = model_dir + '_pruned_' + str(pruning_percentile)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        saver.save(sess, new_dir + '/model.ckpt')
        return new_dir
