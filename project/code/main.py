import numpy as np
import tensorflow as tf
import argparse
import os, tempfile

from baseline import baseline_model_fn
from bayes_mnist import bayes_mnist_model_fn

models = {
    "baseline": baseline_model_fn,
    "bayes_mnist": bayes_mnist_model_fn
}

MNIST_TRAIN_SIZE = 60000

def run(args):

    model_fn = models[args.model]

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=args.model_dir,
                                        params={
                                            "data_format": "channels_last",
                                            "hidden_units": 1200,
                                            "prior": "mixture",
                                            "sigma": 0.,
                                            "mix_prop": 0.25,
                                            "sigma1": 1.,
                                            "sigma2": 6.,
                                            "kl_coeff": "geometric",
                                            # "kl_coeff": "uniform",
                                            "num_batches": MNIST_TRAIN_SIZE // 128,
                                            "optimizer": "sgd",
                                            "learning_rate": 1e-5
                                        })


    tensors_to_log = {"probabilities": "softmax_tensor"}

    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
    #                                           every_n_iter=50)

    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    if args.is_training:
        print("Beginning training of the {} model!".format(args.model))
        classifier.train(input_fn=lambda:mnist_input_fn(train_data, train_labels))
        print("Training finished!")
        # hooks=[logging_hook])

    if args.prune_weights:
        print("Pruning weights.")

    eval_results = classifier.evaluate(input_fn=lambda:mnist_input_fn(eval_data, eval_labels)

    print(eval_results)

def mnist_input_fn(data, labels, num_epochs=10, batch_size=128, shuffle_samples=5000):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(mnist_parse_fn)
    dataset = dataset.batch(batch_size)

    return dataset

def mnist_parse_fn(data, labels):
    return (tf.cast(data, tf.float32)/126., tf.cast(labels, tf.int32))



def is_valid_file(parser, arg):
    """
    Taken from
    https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
    and
    https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
    """
    arg = str(arg)
    if os.path.exists(arg):
        return arg

    dirname = os.path.dirname(arg) or os.getcwd()
    try:
        with tempfile.TemporaryFile(dir=dirname): pass
        return arg
    except Exception:
        parser.error("A file at the given path cannot be created: " % arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--model', choices=list(models.keys()), default='baseline',
                    help='The model to train.')
    parser.add_argument('--no_training', action="store_false", dest="is_training", default=True,
                    help='Should we just evaluate?')
    parser.add_argument('--model_dir', type=lambda x: is_valid_file(parser, x), default='/tmp/bayes_by_backprop',
                    help='The model directory.')
    parser.add_argument('--prune_weights', action="store_true", dest="prune_weights", default=False,
                    help='Should we do weight pruning without training.')
    args = parser.parse_args()

    run(args)
