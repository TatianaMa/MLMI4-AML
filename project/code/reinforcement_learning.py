import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from utils import is_valid_file, load_mushroom_dataset, generate_new_contexts

from baseline_rl_agent import baseline_rl_agent_model_fn

models = {
    "baseline": baseline_rl_agent_model_fn,
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

    # Attach one-hot encoding of actions at the end of context vector
    no_eat_action = np.hstack([context, np.array([1, 0])])
    eat_action = np.hstack([context, np.array([0, 1])])

    actions = np.vstack([no_eat_action, eat_action]).astype(np.float32)

    rewards = agent.predict(input_fn=lambda: tf.data.Dataset.from_tensor_slices(actions))
    rewards = list(rewards)

    # Pick epsilon-greedily
    if np.random.uniform(low=0., high=1.) < epsilon:
        pass

    else:
        action = np.argmax(rewards)

    return action, rewards[action]


def update_agent(agent, features, rewards):
    agent.train(input_fn=lambda:rl_input_fn(features, rewards))


def run(args):

    config = {
        "training_set_size": 8124,
        "num_epochs": 64,
        "batch_size": 64,
        "update_every": 4096,
        "max_steps": 20000,
        "context_size": 112
    }

    model_fn = models[args.model]

    # Create agent
    agent = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=args.model_dir,
                                   params={
                                       "context_size": config["context_size"],
                                       "hidden_units": 100,
                                       "dropout": 0.5,
                                       "num_mc_samples": 1,
                                       "prior": "mixture",
                                       "sigma": 0.,
                                       "mu":0.,
                                       "mix_prop": 0.25,
                                       "sigma1": 6.,
                                       "sigma2": 1.,
                                       #"kl_coeff": "geometric",
                                       "kl_coeff_decay_rate": 1000,
                                       "kl_coeff": "uniform",
                                       "optimizer": "sgd",
                                       "learning_rate": 1e-3
                                   })

    # Load the UCI mushroom dataset
    dataset = load_mushroom_dataset()

    data, oracle_reward, oracle_actions = generate_new_contexts(
        dataset=dataset,
        num_contexts=config["training_set_size"]
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

    rewards = []
    features = None

    cumulative_reward = 0
    cum_rewards = []

    cumulative_regret = 0
    cum_regrets = []

    while True:

        # Shuffle the training set and iterate through it
        order = np.arange(config["training_set_size"], dtype=np.int32)
        np.random.shuffle(order)

        for idx in order:
            if steps == config["max_steps"]: break

            context = contexts[idx, :]
            action, _ = get_action(agent, context)

            # Assign the reward: 0 - pass, 1 - eat the mushroom
            if action == 0:
                reward = no_eat_reward[idx]
            else:
                reward = eat_reward[idx]

            cumulative_reward += reward[0]
            cum_rewards.append(cumulative_reward)

            cumulative_regret += oracle_reward[idx] - reward[0]
            cum_regrets.append(cumulative_regret)

            action_vec = np.zeros((1, 2))
            action_vec[0][action] = 1
            feature_vec = np.hstack([context.reshape((1,) + context.shape), action_vec])

            if features is None:
                features = feature_vec
            else:
                features = np.vstack([features, feature_vec])

            rewards.append(reward)

            # Update the agent's value function
            if steps % config["update_every"] == 0:
                update_agent(agent, features, np.array(rewards))

                # After the update finishes, reset memory
                features = None
                rewards =  []

            if steps % 500 == 0:
                print("{}/{} steps done!".format(steps, config["max_steps"]))
                with open("cum_regrets.txt", "w") as f:
                    f.write(str(cum_regrets))

            steps += 1

        # If we finish iterating through the data, then the else condition will be
        # activated and we will loop around. If we break out of the for loop, it must
        # mean therefore that we have reached the max step count, and we should stop
        # training.
        else:
            continue
        break

    plt.plot(cum_regrets)
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
