import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

def main(args):

    all_ys = []

    for f in args.files:
        res = json.load(f)

        xs = range(1, len(res["tp"]) * args.steps + 1, args.steps)

        fps = []
        current_fp = 0

        fns = []
        current_fn = 0

        for i in range(len(xs)):
            current_fp += res["fp"][i]
            current_fn += res["fn"][i]

            fps.append(current_fp)
            fns.append(current_fn)

        ys = list(map(lambda x: x[0] * 5 + x[1] * 35, zip(fns, fps)))

        all_ys.append(np.array(ys))

        print(ys)

    for i in range(len(all_ys)):
        plt.plot(xs, 10 * all_ys[i], label=args.labels[i])


    plt.yscale("log")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim([3 * 10**4, 2* 10**6])
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Step")
    fig = plt.gcf()
    fig.set_size_inches(6,4.5)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--files', nargs="+", type=open)
    parser.add_argument('--labels', nargs="+", type=str)
    parser.add_argument('--steps', type=int, default = 1)

    args = parser.parse_args()

    main(args)
