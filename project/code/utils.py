import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os, tempfile


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


def setup_eager_checkpoints_and_restore(variables, checkpoint_dir, checkpoint_name="_ckpt"):
    ckpt_prefix = os.path.join(checkpoint_dir, checkpoint_name)

    checkpoint = tf.train.Checkpoint(**{v.name: v for v in variables})

    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_path is None:
        print("No checkpoint found!")
    else:
        print("Checkpoint found at {}, restoring...".format(latest_checkpoint_path))
        checkpoint.restore(latest_checkpoint_path).assert_consumed()
        print("Model restored!")

    return checkpoint, ckpt_prefix

def load_mushroom_dataset():
    """
    7. Attribute Information: (classes: edible=e, poisonous=p)
        1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                    knobbed=k,sunken=s
        2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
        3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                    pink=p,purple=u,red=e,white=w,yellow=y
        4. bruises?:                 bruises=t,no=f
        5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                    musty=m,none=n,pungent=p,spicy=s
        6. gill-attachment:          attached=a,descending=d,free=f,notched=n
        7. gill-spacing:             close=c,crowded=w,distant=d
        8. gill-size:                broad=b,narrow=n
        9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                    green=r,orange=o,pink=p,purple=u,red=e,
                                    white=w,yellow=y
        10. stalk-shape:              enlarging=e,tapering=t
        11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                    rhizomorphs=z,rooted=r,missing=?
        12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
        13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
        14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                    pink=p,red=e,white=w,yellow=y
        15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                    pink=p,red=e,white=w,yellow=y
        16. veil-type:                partial=p,universal=u
        17. veil-color:               brown=n,orange=o,white=w,yellow=y
        18. ring-number:              none=n,one=o,two=t
        19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                    none=n,pendant=p,sheathing=s,zone=z
        20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                    orange=o,purple=u,white=w,yellow=y
        21. population:               abundant=a,clustered=c,numerous=n,
                                    scattered=s,several=v,solitary=y
        22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                    urban=u,waste=w,woods=d
    """
    data_path = tf.keras.utils.get_file("agaricus-lepiota.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data")

    column_names = ["class", "cap-shape", "cap-surface", "cap-color",
                    "bruises?", "odor", "gill-attachment", "gill-spacing",
                    "gill-size", "gill-color", "stalk-shape", "stalk-root",
                    "stalk-surface-above-ring", "stalk-surface-below-ring",
                    "stalk-color-above-ring", "stalk-color-below-ring",
                    "veil-type", "veil-color", "ring-number", "ring-type",
                    "spore-print-color", "population", "habitat"]

    raw_dataset = pd.read_csv(data_path, names=column_names,
                              na_values = "?",
                              sep=",", skipinitialspace=True)

    dataset = raw_dataset.copy()

    # print(dataset.tail())

    # Check if there are any NaN values in the dataset
    # print(dataset.isna().sum())

    # Drop stalk root, because it is missing for 25% of the dataset
    dataset.pop('stalk-root')


    # Convert each column into a number
    for column in dataset:
        # convert the column into a one-hot vector
        one_hot = pd.get_dummies(dataset[column], prefix=column, drop_first=False)

        # attach the one-hot vector to the dataset
        dataset = pd.concat([dataset, one_hot], axis=1)

        # drop the original column
        dataset = dataset.drop(column, axis=1)


    return dataset


def generate_new_contexts(dataset,
                          num_contexts=30,
                          edible_reward=5,
                          poisonous_reward=-35,
                          not_eating_reward=0,
                          prob_poison=0.5):
    """
    Generates the RL tuples: (context, action, reward) as well as the oracle actions and rewards
    """

    num_examples, _ = dataset.shape

    # Pick num_context mushrooms randomly
    contexts = dataset.iloc[np.random.choice(num_examples, num_contexts, replace=True), :]

    # Generate rewards
    not_eating_rewards = not_eating_reward * np.ones((num_contexts, 1))

    # Get multipliers from the one-hot vector
    edible_indicator = contexts.iloc[:, 0]
    poisonous_indicator = contexts.iloc[:, 1]

    # R = e * r_e + p * r, r ~ Cat([r_e, r_p], 0.5)
    eating_rewards = edible_indicator * edible_reward + poisonous_indicator * np.random.choice([edible_reward, poisonous_reward], num_contexts, p=[1 - prob_poison, prob_poison])
    eating_rewards = eating_rewards.to_numpy().reshape((num_contexts, 1))

    possible_rewards = np.concatenate([not_eating_rewards, eating_rewards], axis=1)
    # print(possible_rewards)

    oracle_rewards = np.amax(possible_rewards, axis=1).astype(np.float32)
    oracle_actions = np.argmax(possible_rewards, axis=1)

    # print(oracle_rewards)
    # print(oracle_actions)

    total_oracle_reward = np.sum(oracle_rewards)
    # print(total_oracle_reward)

    is_edible = contexts.iloc[:, 0] == 1

    return (contexts.iloc[:, 2:].to_numpy().astype(np.float32),
            not_eating_rewards.astype(np.float32),
            eating_rewards.astype(np.float32)), oracle_rewards, oracle_actions, is_edible.to_numpy()

if __name__ == "__main__":
    ds = load_mushroom_dataset()
    stuff = generate_new_contexts(ds, 10)
    (ctx, ner, er), ore, ora = stuff

    print(ctx)

    print(er)
    print(ore)
    print(ora)
