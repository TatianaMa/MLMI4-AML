import numpy as np
import tensorflow as tf
import argparse

from baseline import baseline_model_fn
from bayes_mnist import bayes_mnist_model_fn

models = {
    "baseline": baseline_model_fn,
    "bayes_mnist": bayes_mnist_model_fn
}

def run(args):

    model_fn = models[args.model]

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir="/tmp/mnist_baseline_model",
                                        params={
                                            "data_format": "channels_last",
                                            "hidden_units": 800
                                        })


    tensors_to_log = {"probabilities": "softmax_tensor"}

    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
    #                                           every_n_iter=50)

    # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
    #                                                     y=train_labels,
    #                                                     batch_size=100,
    #                                                     num_epochs=None,
    #                                                     shuffle=True)

    print("Beginning training of the {} model!".format(args.model))

    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    classifier.train(input_fn=lambda:mnist_input_fn(train_data, train_labels))
                     # hooks=[logging_hook])

    print("Training finished!")

    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":eval_data},
    #                                                    y=eval_labels,
    #                                                    num_epochs=1,
    #                                                    shuffle=False)

    eval_results = classifier.evaluate(input_fn=lambda:mnist_input_fn(eval_data, eval_labels))

    print(eval_results)

def mnist_input_fn(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(20)
    dataset = dataset.map(mnist_parse_fn)
    dataset = dataset.batch(100)

    return dataset

def mnist_parse_fn(data, labels):
    return (tf.cast(data, tf.float32)/126., tf.cast(labels, tf.int32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CV ML miniproject.')

    parser.add_argument('--model', choices=list(models.keys()), default='baseline',
                    help='The model to train.')
    parser.add_argument('--checkpoint', type=str,
                    help='Path to the checkpoint from which we would like to restore the progress.')

    args = parser.parse_args()

    run(args)
