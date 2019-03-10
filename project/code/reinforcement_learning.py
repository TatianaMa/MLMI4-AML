import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import json

from utils import is_valid_file, load_mushroom_dataset, generate_new_contexts

from baseline_rl_agent import baseline_rl_agent_model_fn
from bayes_rl_agent import bayes_rl_agent_model_fn

models = {
    "baseline": baseline_rl_agent_model_fn,
    "bayes": bayes_rl_agent_model_fn,
}

def rl_input_fn(contexts, rewards, num_epochs=1, batch_size=64, shuffle_size=1000):
    ds = tf.data.Dataset.from_tensor_slices((contexts, rewards))
    ds = ds.shuffle(shuffle_size)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size)

    return ds


def get_action(agent, context, epsilon=0):
    """
    Get the next action as an index (beginning at 0) based on the agent
    and the context vector.

    :param agent: The agent exploring the system
    :type agent: TF Estimator

    :param context: Context vector from the UCI mushrooms dataset
    :type context: [context_size x 1] numpy array
    """

    num_contexts = context.shape[0]

    # Attach one-hot encoding of actions at the end of context vector
    no_eat_action = np.hstack([context, np.ones((num_contexts, 1)), np.zeros((num_contexts, 1))])
    eat_action = np.hstack([context, np.zeros((num_contexts, 1)), np.ones((num_contexts, 1))])
    no_eat_rewards = agent.predict(input_fn=lambda: tf.data.Dataset.from_tensor_slices(no_eat_action))
    no_eat_rewards = np.array(list(no_eat_rewards))

    eat_rewards = agent.predict(input_fn=lambda: tf.data.Dataset.from_tensor_slices(eat_action))
    eat_rewards = np.array(list(eat_rewards))

    rewards = np.hstack([no_eat_rewards, eat_rewards])

    # Epsilon-greedy policy
    # Start completely greedy
    action = np.argmax(rewards, axis=1)

    # Select indices to update
    rand_indices = np.random.uniform(low=0., high=1., size=num_contexts) < epsilon

    # Select random actions
    rand_actions = np.random.choice([0, 1], size=num_contexts)

    action[rand_indices] = rand_actions[rand_indices]

    return action


def update_agent(agent, features, rewards):
    agent.train(input_fn=lambda:rl_input_fn(features, rewards))


def run(args):

    if args.eps < 0 or args.eps > 1:
        raise Exception("Epsilon has to be between 0 and 1!")

    config = {
        "training_set_size": 8124,
        "num_epochs": 64,
        "batch_size": 64,
        "replay_buffer_size": 4096,
        "update_every": 20,
        "max_steps": 50000,
        "context_size": 112,
        "num_warmup_batches": 00,
        "log_every": 10
    }

    model_fn = models[args.model]

    num_batches = config["max_steps"] // config["update_every"]

    # Create agent
    agent = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=args.model_dir,
                                   params={
                                       "context_size": config["context_size"],
                                       "hidden_units": 100,
                                       "dropout": 0.5,
                                       "num_mc_samples": 2,
                                       "prior": "mixture",
                                       "sigma": 0.,
                                       "mu":0.,
                                       "mix_prop": 0.25,
                                       "sigma1": 6.,
                                       "sigma2": 1.,
                                       #"kl_coeff": "geometric",
                                       "kl_coeff_decay_rate": 1000,
                                       "kl_coeff": "uniform",
                                       "num_batches": num_batches,
                                       "optimizer": "sgd",
                                       "learning_rate": 1e-3,
                                       "verbose": False
                                   })

    # Load the UCI mushroom dataset
    dataset = load_mushroom_dataset()

    data, oracle_reward, oracle_actions, is_edible = generate_new_contexts(
        dataset=dataset,
        num_contexts=config["max_steps"]
    )

    contexts, no_eat_reward, eat_reward = data

    # ==========================================================================
    # Perform task
    # ==========================================================================

    # We count how many times we performed each action. This is used later for
    # statistics, but more importantly, it is used to generate some initial data
    # by forcing the agent to explore a bit
    action_counter = np.zeros((2, 1))

    steps = 1

    rewards = None
    replay_buffer = None

    cumulative_reward = 0
    cum_rewards = []

    cumulative_regret = 0
    cum_regrets = []

    oracle_relative_action_taken = {
        "tp": [],
        "fp": [],
        "fn": [],
        "tn": []
    }

    edibility_relative_action_taken = {
        "tp": [],
        "fp": [],
        "fn": [],
        "tn": []
    }

    oracle_stats = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0
    }

    is_edible_stats = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0
    }


    batch_size = config["update_every"]

    print("Training set size: {}".format(config["training_set_size"]))
    print("Update frequency: {}".format(config["update_every"]))
    print("Number of batches: {}".format(num_batches))
    print("Cumulative oracle reward: {}".format(np.sum(oracle_reward)))

    total_batch_index = 0

    for batch_idx in range(num_batches):

        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        total_batch_index += 1

        context = contexts[ start_idx:end_idx , :]

        # For the first few batches, just sample them randomly
        if total_batch_index <= config["num_warmup_batches"]:
            action = np.random.choice([0, 1], batch_size)
        else:
            action = get_action(agent, context, epsilon=args.eps)

        # Assume we haven't eaten anything, correct where needed
        reward = no_eat_reward[ start_idx: end_idx, :]
        curr_eat_rewards = eat_reward[ start_idx:end_idx, :]
        reward[action == 1] = curr_eat_rewards[action == 1]

        ore = oracle_reward[start_idx:end_idx].reshape((-1, 1))

        cumulative_reward += np.sum(reward)
        cum_rewards.append(cumulative_reward)

        regret = np.sum(ore - reward)
        cumulative_regret += regret
        cum_regrets.append(cumulative_regret)

        action_vec = np.zeros((batch_size, 2))
        action_vec[action == 0, 0] = 1
        action_vec[action == 1, 1] = 1
        feature_vec = np.hstack([context, action_vec])


        if replay_buffer is None:
            replay_buffer = feature_vec
            rewards = reward
        else:
            replay_buffer = np.vstack([replay_buffer, feature_vec])
            rewards = np.vstack([rewards, reward])

        # Prune the replay buffer
        if replay_buffer.shape[0] > config["replay_buffer_size"]:
            replay_buffer = replay_buffer[-config["replay_buffer_size"]:, :]
            rewards = rewards[-config["replay_buffer_size"]:, :]


        # Update the agent's value function
        update_agent(agent, replay_buffer, np.array(rewards))

        oracle_stats["tp"] += sum((action == 1) & (oracle_actions[start_idx:end_idx] == 1))
        oracle_stats["fp"] += sum((action == 1) & (oracle_actions[start_idx:end_idx] == 0))
        oracle_stats["tn"] += sum((action == 0) & (oracle_actions[start_idx:end_idx] == 0))
        oracle_stats["fn"] += sum((action == 0) & (oracle_actions[start_idx:end_idx] == 1))

        is_edible_stats["tp"] += sum((action == 1) & (is_edible[start_idx:end_idx] == 1))
        is_edible_stats["fp"] += sum((action == 1) & (is_edible[start_idx:end_idx] == 0))
        is_edible_stats["tn"] += sum((action == 0) & (is_edible[start_idx:end_idx] == 0))
        is_edible_stats["fn"] += sum((action == 0) & (is_edible[start_idx:end_idx] == 1))


        # ======================================================================
        # Log things
        # ======================================================================
        if total_batch_index % config["log_every"] == 0:
            print("{}/{} batches done!".format(total_batch_index, num_batches))
            with open("cum_regrets_{}_eps_{:.2f}.txt".format(args.model, args.eps), "w") as f:
                f.write(str(cum_regrets))

            oracle_relative_action_taken["tp"].append(int(oracle_stats["tp"]))
            oracle_relative_action_taken["fp"].append(int(oracle_stats["fp"]))
            oracle_relative_action_taken["tn"].append(int(oracle_stats["tn"]))
            oracle_relative_action_taken["fn"].append(int(oracle_stats["fn"]))

            edibility_relative_action_taken["tp"].append(int(is_edible_stats["tp"]))
            edibility_relative_action_taken["fp"].append(int(is_edible_stats["fp"]))
            edibility_relative_action_taken["tn"].append(int(is_edible_stats["tn"]))
            edibility_relative_action_taken["fn"].append(int(is_edible_stats["fn"]))


            oracle_stats = {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0
            }

            is_edible_stats = {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0
            }


            with open("cum_regrets_{}_eps_{:.2f}_orat.txt".format(args.model, args.eps), "w") as f:
                json.dump(oracle_relative_action_taken, f)

            with open("cum_regrets_{}_eps_{:.2f}_erat.txt".format(args.model, args.eps), "w") as f:
                json.dump(edibility_relative_action_taken, f)

        num_incorrect_actions = np.sum(np.abs(action - oracle_actions[ start_idx: end_idx]))
        if num_incorrect_actions == 0:
            print("Perfect set of actions!")
        else:
            print("{}/{} actions selected incorrectly. Regret:{}".format(num_incorrect_actions, batch_size, regret))



    plt.plot(cum_regrets)
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--eps', type=float, default=0.0,
                        help='Epsilon for the Eps-Greedy policy')
    parser.add_argument('--model', choices=list(models.keys()), default='baseline',
                    help='The model to train.')
    parser.add_argument('--no_training', action="store_false", dest="is_training", default=True,
                    help='Should we just evaluate?')
    parser.add_argument('--model_dir', type=lambda x: is_valid_file(parser, x), default='/tmp/bayes_by_backprop',
                    help='The model directory.')

    args = parser.parse_args()

    run(args)
