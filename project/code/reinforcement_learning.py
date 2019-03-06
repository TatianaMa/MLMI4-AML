import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from utils import is_valid_file, load_mushroom_dataset, generate_new_contexts

models = {
    "baseline": None
}

def prepare_rl_task():
    dataset = load_mushroom_dataset()

    generate_new_contexts(dataset)

def run(args):

    prepare_rl_task()


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
