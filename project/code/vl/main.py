import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfe = tf.contrib.eager
tfs = tf.contrib.summary
tfs_logger = tfs.record_summaries_every_n_global_steps

import argparse
import os
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
    return (tf.cast(tf.reshape(data, [-1]), tf.float32)/126., tf.cast(labels, tf.int64))


def run(args):

    # ==========================================================================
    # Configuration
    # ==========================================================================
    config = {
        "training_set_size": 60000,
        "num_epochs": 1,
        "batch_size": 128,
        "pruning_percentile": 70,
        "learning_rate": 1e-3,
        "log_freq": 100,
        "checkpoint_postfix": "_ckpt",
        "validation_set_percentage": 0.1,
    }

    #num_batches = config["training_set_size"] * config["num_epochs"] / config["batch_size"]
    num_batches = int((1 - config["validation_set_percentage"]) * config["training_set_size"]) / config["batch_size"]

    # ==========================================================================
    # Loading in the dataset
    # ==========================================================================
    ((train_data, train_labels),
    (test_data, test_labels)) = tf.keras.datasets.mnist.load_data()

    train_data, val_data, train_labels, val_labels = train_test_split(train_data,
                                                                      train_labels,
                                                                      test_size=config["validation_set_percentage"],
                                                                      shuffle=True,
                                                                      stratify=train_labels)


    train_dataset = mnist_input_fn(train_data,
                                   train_labels,
                                   batch_size=config["batch_size"])

    val_dataset = mnist_input_fn(val_data,
                                 val_labels,
                                 batch_size=len(val_data))


    # ==========================================================================
    # Define the model
    # ==========================================================================

    model = VarMNIST(units=400,
                     prior=tfp.distributions.Normal(loc=0., scale=0.3))
    model(tf.zeros((1, 28, 28)))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=config["learning_rate"])

    # ==========================================================================
    # Define Checkpoints
    # ==========================================================================

    global_step = tf.train.get_or_create_global_step()

    ckpt_prefix = os.path.join(args.model_dir, "checkpoints", config["checkpoint_postfix"])

    checkpoint = tf.train.Checkpoint(**{v.name: v for v in model.get_all_variables() + (global_step,)})

    latest_checkpoint_path = tf.train.latest_checkpoint(os.path.join(args.model_dir, "checkpoints"))

    if latest_checkpoint_path is None:
        print("No checkpoint found!")
    else:
        print("Checkpoint found at {}, restoring...".format(latest_checkpoint_path))
        checkpoint.restore(latest_checkpoint_path).assert_consumed()
        print("Model restored!")

    # ==========================================================================
    # Define Tensorboard Summaries
    # ==========================================================================

    logdir = os.path.join(args.model_dir, "log")
    writer = tfs.create_file_writer(logdir)
    writer.set_as_default()

    train_accuracy = tfe.metrics.Accuracy()
    val_accuracy = tfe.metrics.Accuracy()
    test_accuracy = tfe.metrics.Accuracy()

    for validation_data, validation_labels in val_dataset:
        val_data = validation_data
        val_labels = validation_labels

    # ==========================================================================
    # Train the model
    # ==========================================================================

    for epoch in range(1, config["num_epochs"] + 1):

        with tqdm(total=num_batches) as pbar:
            for features, labels in train_dataset:
                # Increment global step
                global_step.assign_add(1)

                # Record gradients of the forward pass
                with tf.GradientTape() as tape:

                    logits = model(features)

                    negative_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels,
                        logits=logits)
                    negative_log_likelihood = tf.reduce_sum(negative_log_likelihood)

                    kl_coeff = 1. / num_batches

                    # negative ELBO
                    loss = kl_coeff * model.kl_divergence + negative_log_likelihood

                # Backprop
                grads = tape.gradient(loss, model.get_all_variables())
                optimizer.apply_gradients(zip(grads, model.get_all_variables()))

                # =================================
                # Add summaries for tensorboard
                # =================================
                with tfs_logger(config["log_freq"]):
                    tfs.scalar("Loss", loss)

                    predictions = tf.argmax(input=logits,
                                            axis=1)
                    train_accuracy(labels=labels,
                                   predictions=predictions)

                    acc = 100 * train_accuracy.result()

                # Update the progress bar
                pbar.update(1)
                pbar.set_description("Epoch {}, Train Accuracy: {:.2f}, ELBO: {:.2f}".format(epoch, acc, loss))


        logits = model(val_data)
        val_predictions = tf.argmax(input=logits,
                                    axis=1)

        val_accuracy(labels=val_labels,
                     predictions=val_predictions)
        acc = val_accuracy.result()

        print("Validation Accuracy: {:.2f}%".format(100 * acc))

        tfs.scalar("Validation Accuracy", acc)

        checkpoint.save(ckpt_prefix)

    # ==========================================================================
    # Testing
    # ==========================================================================

    # Silly hack to get the entire training set
    for data, labels in mnist_input_fn(test_data, test_labels, batch_size=len(test_data)):
        test_data = data
        test_labels = labels

    logits = model(test_data)
    predictions = tf.argmax(input=logits,
                            axis=1)
    test_accuracy(labels=test_labels,
                  predictions=predictions)

    acc = 100 * test_accuracy.result()
    print("Test accuracy: {:.2f}%".format(acc))

    # ==========================================================================
    # Weight pruning
    # ==========================================================================

    binwidth = 1
    snr_vector = np.log(np.abs(model.mu_vector.numpy())) - np.log(model.sigma_vector)
    snr_vector = 10. * snr_vector

    pruning_threshold = np.percentile(snr_vector,
                                      q=config["pruning_percentile"],
                                      interpolation='lower')

    print("Pruning threshold is {:.2f}".format(pruning_threshold))

    model.prune_below_snr(pruning_threshold)

    logits = model(test_data)
    predictions = tf.argmax(input=logits,
                            axis=1)
    test_accuracy(labels=test_labels,
                  predictions=predictions)

    acc = 100 * test_accuracy.result()
    print("Pruned {:.2f}% of weights. Accuracy: {}%".format(
        config["pruning_percentile"],
        acc))

    plt.hist(snr_vector, bins=np.arange(min(snr_vector), max(snr_vector) + binwidth, binwidth))
    plt.axvline(x=pruning_threshold, color='tab:red')
    plt.xlabel('Signal-To-Noise Ratio (dB)')
    plt.ylabel('Density')
    plt.title('Histogram of the Signal-To-Noise ratio over all weights in the network')
    plt.show()


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

