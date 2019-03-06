# Adapted from https://github.com/tensorflow/tensorflow/issues/16646#issuecomment-384935839
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def prune_weights(model_dir, pruning_percentile, plot_hist=False):
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

        if plot_hist:
            binwidth = 1
            snrs_val = sess.run(snrs)
            plt.hist(snrs_val, bins=np.arange(min(snrs_val), max(snrs_val) + binwidth, binwidth))
            plt.axvline(x=pruning_threshold, color='tab:red')
            plt.xlabel('Signal-To-Noise Ratio (dB)')
            plt.ylabel('Density')
            plt.title('Histogram of the Signal-To-Noise ratio over all weights in the network')
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
                mask = tf.dtypes.cast(tf.math.greater(snr, pruning_threshold), dtype=tf.float32)
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
        return new_dir
