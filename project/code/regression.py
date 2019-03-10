import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from utils import is_valid_file

from baseline_regression import baseline_regression_model_fn
from bayes_regression import bayes_regression_model_fn

models = {
    "baseline_regression": baseline_regression_model_fn,
    "bayes_regression": bayes_regression_model_fn,
}


def create_sine_training_data(num_examples=200):
    xs = np.random.uniform(low=-0.05, high=0.6, size=num_examples)
    eps = np.random.normal(loc=0., scale=0.02, size=[num_examples])

    ys = xs + 0.3 * np.sin(2*np.pi * (xs + eps)) + 0.3 * np.sin(4*np.pi * (xs + eps)) + eps

<<<<<<< HEAD
=======
    xs = xs
    ys = ys

>>>>>>> 5ea20dc8a593374eb6078c0903ce4084c28267fc
    return xs, ys


def regression_input_fn(training_xs,
                        training_ys,
                        num_epochs=100,
                        batch_size=1):
                        #shuffle_samples=1000

    dataset = tf.data.Dataset.from_tensor_slices((training_xs.astype(np.float32), training_ys.astype(np.float32)))
    #dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset


def run(args):

    config = {
<<<<<<< HEAD
        "training_set_size": 200,
        "num_epochs": 400,
=======
        "training_set_size": 300,
        "num_epochs": 150,
>>>>>>> 5ea20dc8a593374eb6078c0903ce4084c28267fc
        "batch_size": 1,
    }

    num_batches = config["training_set_size"] / config["batch_size"]

    print("Number of batches: {}".format(num_batches))

    model_fn = models[args.model]

    regressor = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=args.model_dir,
                                        params={
                                            "data_format": "channels_last",
                                            "input_dims": [1],
                                            "output_dims": 1,
<<<<<<< HEAD
                                            "hidden_units": 100,
                                            "num_mc_samples": 1,
                                            #"prior": "gaussian",
=======
                                            "hidden_units": 400,
                                            "num_mc_samples": 1,
>>>>>>> 5ea20dc8a593374eb6078c0903ce4084c28267fc
                                            "prior": "mixture",
                                            "sigma": 2.,
                                            "mu":0.,
                                            "mix_prop": 0.25,
                                            "sigma1": 6.,
                                            "sigma2": 1.,
                                            #"kl_coeff": "geometric",
                                            "kl_coeff_decay_rate": 100,
                                            "kl_coeff": "uniform",
                                            "num_batches": num_batches,
                                            "optimizer": "adam",
                                            "learning_rate": 1e-3
                                        })


    training_xs, training_ys = create_sine_training_data(
        num_examples=config["training_set_size"],
        )

    if args.is_training:
        print("Beginning training of the {} model!".format(args.model))

        regressor.train(input_fn=lambda:regression_input_fn(training_xs,
                                                            training_ys,
                                                            num_epochs=config["num_epochs"],
                                                            batch_size=config["batch_size"]))
        print("Training finished!")

<<<<<<< HEAD
    xs = np.linspace(start=-2., stop=2., num=400)

    results_overall = []
    #mus_overall = []
    #sigmas_overall = []
=======
    xs = np.linspace(start=-0.2, stop=1.2, num=100)

    mus_overall = []
    sigmas_overall = []
    results_overall = []
>>>>>>> 5ea20dc8a593374eb6078c0903ce4084c28267fc

    for i in range(100):
        results = regressor.predict(
            input_fn=lambda: tf.data.Dataset.from_tensor_slices(xs.astype(np.float32)).batch(1))

<<<<<<< HEAD
        results_overall.append([r[0] for r in results])
        #mus, sigmas = zip(*list(results))
        #mus_overall.append(mus)
        #sigmas_overall.append(sigmas)

    results_overall = np.array(results_overall)
    #mus_overall = np.array(mus_overall)
    #sigmas_overall = np.array(sigmas_overall)

    means = np.median(results_overall, axis=0)
    bottom_25 = np.percentile(results_overall, 25, axis=0)
    top_25 = np.percentile(results_overall, 75, axis=0)
    bottom_25_2 = np.percentile(results_overall, 0, axis=0)
    top_25_2 = np.percentile(results_overall, 100, axis=0)
    
    #means = np.median(mus_overall, axis=0)
    #sigmas = np.median(sigmas_overall, axis=0)
    #bottom_25 = np.percentile(mus_overall, 25, axis=0)
    #top_25 = np.percentile(mus_overall, 75, axis=0)

    plt.plot(xs, means)
    #plt.plot(xs, means + sigmas)
    #plt.plot(xs, means - sigmas)
=======
        # mus, sigmas = zip(*list(results))
        # mus_overall.append(mus)
        # sigmas_overall.append(sigmas)
        results_overall.append([r[0] for r in results])

    results_overall = np.array(results_overall)
    print(results_overall.shape)
    # mus_overall = np.array(mus_overall)
    # sigmas_overall = np.array(sigmas_overall)

    # mus = np.median(mus_overall, axis=0)
    # sigmas = np.median(sigmas_overall, axis=0)
    results = np.median(results_overall, axis=0)
    bottom_25 = np.percentile(results_overall, 25, axis=0)
    top_25 = np.percentile(results_overall, 75, axis=0)

    plt.plot(xs, results)
    # plt.plot(xs, means + sigmas)
    # plt.plot(xs, means - sigmas)
>>>>>>> 5ea20dc8a593374eb6078c0903ce4084c28267fc
    plt.plot(xs, bottom_25, color='r')
    plt.plot(xs, top_25, color='r')
    plt.plot(xs, bottom_25_2, color='g')
    plt.plot(xs, top_25_2, color='g')
    plt.scatter(training_xs, training_ys, marker='x', color='k')
    plt.ylim([-1.5,1.9])
    plt.xlim([-0.6,1.4])
    plt.show()


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
