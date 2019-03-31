import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfe = tf.contrib.eager
tfs = tf.contrib.summary
tfs_logger = tfs.record_summaries_every_n_global_steps

import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import json

from utils import is_valid_file, setup_eager_checkpoints_and_restore
from variational import VarRegression

tf.enable_eager_execution()


models = {
    "baseline": None,
    "bayes": VarRegression,
}


def create_sine_training_data(num_examples=200):
    np.random.seed(1)
    xs = np.random.uniform(low=-0.05, high=0.6, size=num_examples)
    eps = np.random.normal(loc=0., scale=0.02, size=[num_examples])

    ys = xs + 0.3 * np.sin(2*np.pi * (xs + eps)) + 0.3 * np.sin(4*np.pi * (xs + eps)) + eps

    return xs, ys


def regression_input_fn(training_xs,
                        training_ys,
                        batch_size=1,
                        shuffle_samples=1000):

    dataset = tf.data.Dataset.from_tensor_slices((training_xs.astype(np.float32), training_ys.astype(np.float32)))
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.batch(batch_size)

    return dataset


def run(args):

    # ==========================================================================
    # Configuration
    # ==========================================================================
    config = {
        "training_set_size": 200,
        "num_epochs": 10,
        "batch_size": 1,
        "num_units": 400,
        "checkpoint_name": "_ckpt",
        "learning_rate": 1e-3,
        "log_freq": 100,
    }

    num_batches = config["training_set_size"] / config["batch_size"]

    print("Number of batches: {}".format(num_batches))

    # ==========================================================================
    # Loading in the dataset
    # ==========================================================================

    training_xs, training_ys = create_sine_training_data(
        num_examples=config["training_set_size"],
        )

    train_dataset = regression_input_fn(training_xs,
                                        training_ys,
                                        batch_size=config["batch_size"])

    # ==========================================================================
    # Define the model
    # ==========================================================================

    model = models[args.model](units=config["num_units"],
                               prior=tfp.distributions.Normal(loc=0., scale=0.3))

    # Connect the model computational graph by executing a forward-pass
    model(tf.zeros((1, 1)))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=config["learning_rate"])

    # ==========================================================================
    # Define Checkpoints
    # ==========================================================================
    global_step = tf.train.get_or_create_global_step()

    trainable_vars = model.get_all_variables() + (global_step,)
    checkpoint_dir = os.path.join(args.model_dir, "checkpoints")

    checkpoint, ckpt_prefix = setup_eager_checkpoints_and_restore(
        variables=trainable_vars,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=config["checkpoint_name"])

    # ==========================================================================
    # Define Tensorboard Summaries
    # ==========================================================================

    # ==========================================================================
    # Train the model
    # ==========================================================================

    if args.is_training:
        for epoch in range(1, config["num_epochs"] + 1):

            with tqdm(total=num_batches) as pbar:
                for xs, ys in train_dataset:
                    # Increment global step
                    global_step.assign_add(1)

                    # Record gradients of the forward pass
                    with tf.GradientTape() as tape:

                        logits = model(xs)

                        kl_coeff = 1. / num_batches

                        # negative ELBO
                        loss = kl_coeff * model.kl_divergence + model.negative_log_likelihood(logits, tf.reshape(ys, [-1, 1]))

                    # Backprop
                    grads = tape.gradient(loss, model.get_all_variables())
                    optimizer.apply_gradients(zip(grads, model.get_all_variables()))

                    # =================================
                    # Add summaries for tensorboard
                    # =================================
                    with tfs_logger(config["log_freq"]):
                        tfs.scalar("Loss", loss)

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_description("Epoch {}, ELBO: {:.2f}".format(epoch, loss))


            # logits = model(val_data)
            # val_predictions = tf.argmax(input=logits,
            #                             axis=1)

            # val_accuracy(labels=val_labels,
            #             predictions=val_predictions)
            # acc = val_accuracy.result()

            # print("Validation Accuracy: {:.2f}%".format(100 * acc))

            # tfs.scalar("Validation Accuracy", acc)

            checkpoint.save(ckpt_prefix)

    else:
        print("Skipping training!")


    # ==========================================================================
    # Testing
    # ==========================================================================

    xs = np.linspace(start=-2., stop=2, num=400)

    mus_overall = []
    sigmas_overall = []
    results_overall = []

    for i in range(10):
        results = model(tf.convert_to_tensor(xs, dtype=tf.float32))

        results_overall.append(results.numpy())

    results_overall = np.array(results_overall)

    means = np.median(results_overall, axis=0)
    bottom_25 = np.percentile(results_overall, 25, axis=0)
    top_25 = np.percentile(results_overall, 75, axis=0)
    bottom_25_2 = np.percentile(results_overall, 0, axis=0)
    top_25_2 = np.percentile(results_overall, 100, axis=0)

    fig = plt.gcf()
    fig.set_size_inches(5,3.5)
    plt.plot(xs, means)
    plt.plot(xs, bottom_25, color='r')
    plt.plot(xs, top_25, color='r')
    plt.plot(xs, bottom_25_2, color='g')
    plt.plot(xs, top_25_2, color='g')
    plt.scatter(training_xs, training_ys, marker='x', color='k')
    plt.ylim([-1.5,1.5])
    plt.xlim([-0.6,1.4])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--model', choices=list(models.keys()), default='bayes',
                    help='The model to train.')
    parser.add_argument('--no_training', action="store_false", dest="is_training", default=True,
                    help='Should we just evaluate?')
    parser.add_argument('--model_dir', type=lambda x: is_valid_file(parser, x), default='/tmp/bayes_by_backprop_regression',
                    help='The model directory.')

    args = parser.parse_args()

    run(args)
