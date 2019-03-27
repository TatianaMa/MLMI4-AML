import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfe = tf.contrib.eager
tfs = tf.contrib.summary
tfs_logger = tfs.record_summaries_every_n_global_steps

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import is_valid_file
from variational import VarMNIST

tf.enable_eager_execution()

models = {
    "baseline": None
}

def mnist_input_fn(data, labels, batch_size=128, shuffle_samples=5000):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.map(mnist_parse_fn)
    dataset = dataset.batch(batch_size)

    return dataset


def mnist_parse_fn(data, labels):#shuffle_samples=5000
    return (tf.cast(tf.reshape(data, [-1]), tf.float32)/126., tf.cast(labels, tf.int32))


def run(args):

    # ==========================================================================
    # Configuration
    # ==========================================================================
    config = {
        "training_set_size": 60000,
        "num_epochs": 10,
        "batch_size": 128,
        "pruning_percentile": 98,
        "learning_rate": 1e-3,
        "log_freq": 100
    }

    #num_batches = config["training_set_size"] * config["num_epochs"] / config["batch_size"]
    num_batches = config["training_set_size"] / config["batch_size"]

    # ==========================================================================
    # Loading in the dataset
    # ==========================================================================
    ((train_data, train_labels),
    (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    dataset = mnist_input_fn(train_data,
                             train_labels,
                             batch_size=config["batch_size"])

    # ==========================================================================
    # Define the model
    # ==========================================================================

    global_step = tf.train.get_or_create_global_step()

    model = VarMNIST(units=400,
                     prior=tfp.distributions.Normal(loc=0., scale=0.3))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=config["learning_rate"])

    # ==========================================================================
    # Train the model
    # ==========================================================================

    for epoch in range(1, config["num_epochs"] + 1):

        with tqdm(total=num_batches) as pbar:
            for features, labels in dataset:
                # Increment global step
                global_step.assign_add(1)

                # Record gradients of the forward pass
                with tf.GradientTape() as tape, tfs_logger(config["log_freq"]):

                    logits = model(features)

                    negative_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels,
                        logits=logits)
                    negative_log_likelihood = tf.reduce_sum(negative_log_likelihood)

                    kl_coeff = 1. / num_batches

                    # negative ELBO
                    loss = kl_coeff * model.kl_divergence + negative_log_likelihood

                    # Add tensorboard summaries
                    tfs.scalar("Loss", loss)

                # Backprop
                grads = tape.gradient(loss, model.get_all_variables())
                optimizer.apply_gradients(zip(grads, model.get_all_variables()))

                # Update the progress bar
                pbar.update(1)
                pbar.set_description("Epoch {}, ELBO: {:.2f}".format(epoch, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--model', choices=list(models.keys()), default='baseline',
                    help='The model to train.')
    parser.add_argument('--no_training', action="store_false", dest="is_training", default=True,
                    help='Should we just evaluate?')
    parser.add_argument('--model_dir', type=lambda x: is_valid_file(parser, x), default='/tmp/bayes_by_backprop',
                    help='The model directory.')
    parser.add_argument('--prune_weights', action="store_true", dest="prune_weights", default=False,
                    help='Should we do weight pruning during evaluation.')
    args = parser.parse_args()

    run(args)

