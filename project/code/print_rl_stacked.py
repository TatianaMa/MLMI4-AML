import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import seaborn as sns

def main(args):
    res = json.load(args.file)

    #colours = sns.cubehelix_palette(4, start=.5, rot=-.75)

    #colours = [colours[i] for i in [2, 0, 3, 1]]
    colours = sns.color_palette("RdBu", 4)
    colours = [colours[i] for i in [0, 1, 3, 2]]

    #labels = ["True Positives", "False Positives", "True Negatives", "False Negatives"]
    labels = ["Agent eats edible mushroom", "Agent eats poisonous mushroom", "Agent passes on poisonous mushroom", "Agent passes on edible mushroom"]
    xs = range(1, len(res["tp"][:50]) * args.steps + 1, args.steps)

    ys = [np.array(res[k][:50]) / float(args.steps) * 100. for k in ["tp", "fp", "tn", "fn"]]

    plt.stackplot(xs, ys, labels=labels, colors=colours)
    plt.legend(loc='lower right', frameon=True, framealpha=0.3)
    plt.xlim([1, (len(xs) - 1) * args.steps])
    plt.ylim([0, 100])
    plt.xlabel("Step")
    plt.ylabel("Outcome (%)")
    fig = plt.gcf()
    fig.set_size_inches(6,4.5)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--file', type=open)
    parser.add_argument('--steps', type=int, default = 1)

    args = parser.parse_args()

    main(args)
