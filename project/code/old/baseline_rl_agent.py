import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

NUM_ACTIONS = 2

def create_model(features, params):

    input_layer = tf.cast(tf.reshape(features, [-1, params["context_size"] + NUM_ACTIONS]), tf.float64)

    # Dense Layer #1
    dense1 = Dense(
        units=params['hidden_units'],
        activation=tf.nn.relu
    ).apply(input_layer)

    #dense1 = Dropout(0.5).apply(dense1)

    # Dense Layer #2
    dense2 = Dense(
        units=params['hidden_units'],
        activation=tf.nn.relu
    ).apply(dense1)

    #dense2 = Dropout(0.3).apply(dense2)

    # Output Layer
    expected_reward = Dense(
        units=1
    ).apply(dense2)

    return expected_reward


def baseline_rl_agent_model_fn(features, labels, mode, params):

    if "learning_rate" not in params:
        raise KeyError("No learning rate specified!")

    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]),
        "momentum": tf.train.MomentumOptimizer(learning_rate=params["learning_rate"], momentum=0.9, use_nesterov=True),
        "adam": tf.train.AdamOptimizer(learning_rate=params["learning_rate"]),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])
    }

    expected_reward = create_model(features, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=expected_reward)

    loss = tf.losses.mean_squared_error(predictions=expected_reward,
                                        labels=labels)


    if mode == tf.estimator.ModeKeys.TRAIN:
        try:
            optimizer = optimizers[params["optimizer"]]
        except KeyError as e:
            raise KeyError("No optimizer specified! Possibilities: {}".format(optimizers.keys()))

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            "MSE": loss
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
