import numpy as np
import tensorflow as tf
import argparse
import os, tempfile
import matplotlib.pyplot as plt

from baseline import baseline_model_fn
from baseline_regression import baseline_regression_model_fn
from bayes_mnist import bayes_mnist_model_fn
from bayes_regression import bayes_regression_model_fn



MNIST_TRAIN_SIZE = 60000

def run(args):

    input_fn, model_fn = models[args.model]

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=args.model_dir,
                                        params={
                                            "data_format": "channels_last",
                                            "input_dims": [1],
                                            "output_dims": 1,
                                            "hidden_units": 60,
                                            "prior": "mixture",
                                            "sigma": 0.,
                                            "mu":0.,
                                            "mix_prop": 0.25,
                                            "sigma1": 1.,
                                            "sigma2": 6.,
                                            #"kl_coeff": "geometric",
                                            "kl_coeff_decay_rate": 1000,
                                            "kl_coeff": "uniform",
                                            "num_batches": MNIST_TRAIN_SIZE // 128,
                                            "optimizer": "rmsprop",
                                            "learning_rate": 1e-3
                                        })


    tensors_to_log = {"probabilities": "softmax_tensor"}

    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
    #                                           every_n_iter=50)



    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    if args.model == "bayes_regression" or args.model == "baseline_regression":
        train_xs, train_ys, input_fn = create_sine_regression_input_fn()


    if args.is_training:
        print("Beginning training of the {} model!".format(args.model))
        classifier.train(input_fn=lambda:input_fn(train_data, train_labels)) #input_fn(train_data, train_labels)
        print("Training finished!")
        # hooks=[logging_hook])

    xs = np.linspace(start=-1, stop=2, num=100)
    #preds = [p for p in classifier.predict(tf.data.Dataset.from_tensor_slices(tf.cast(xs, tf.float32)))]
    #    results = classifier.predict(input_fn=tf.data.Dataset.from_tensor_slices(tf.cast(xs, tf.float32)))

    results_overall = []

    for i in range(20):
        results = classifier.predict(input_fn=lambda: tf.data.Dataset.from_tensor_slices(xs.astype(np.float32)).batch(1))
        results_overall.append([r[0] for r in results])

    results_overall = np.array(results_overall)

    means = np.median(results_overall, axis=0)
    bottom_25 = np.percentile(results_overall, 25, axis=0)
    top_25 = np.percentile(results_overall, 75, axis=0)

    plt.plot(xs, means)
    #plt.plot(xs, bottom_25, color='r')
    #plt.plot(xs, top_25, color='r')
    plt.scatter(train_xs, train_ys, marker='x', color='k')
    plt.show()

    #print(eval_results)

def mnist_input_fn(data, labels, num_epochs=10, batch_size=128, shuffle_samples=5000):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(mnist_parse_fn)
    dataset = dataset.batch(batch_size)

    return dataset

def create_sine_regression_input_fn(num_examples=500, num_epochs=100000, batch_size=250, shuffle_samples=1000):
    xs = np.random.uniform(low=0., high=0.6, size=num_examples)
    eps = np.random.normal(loc=0., scale=0.02, size=[num_examples])

    ys = xs + 0.3 * np.sin(2*np.pi * (xs + eps)) + 0.3 * np.sin(4*np.pi * (xs + eps)) + eps

    def input_fn(d, l):
        dataset = tf.data.Dataset.from_tensor_slices((xs.astype(np.float32), ys.astype(np.float32)))
        dataset = dataset.shuffle(shuffle_samples)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        return dataset

    return xs, ys, input_fn


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


models = {
    "baseline": (mnist_input_fn, baseline_model_fn),
    "bayes_mnist": (mnist_input_fn, bayes_mnist_model_fn),
    "baseline_regression": (create_sine_regression_input_fn, baseline_regression_model_fn),
    "bayes_regression": (create_sine_regression_input_fn, bayes_regression_model_fn),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--model', choices=list(models.keys()), default='baseline',
                    help='The model to train.')
    parser.add_argument('--no_training', action="store_false", dest="is_training", default=True,
                    help='Should we just evaluate?')
    parser.add_argument('--model_dir', type=lambda x: is_valid_file(parser, x), default='/tmp/bayes_by_backprop',
                    help='The model directory.')

    args = parser.parse_args()

    run(args)
