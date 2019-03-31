import tensorflow as tf


def create_model(features, params):
    """
    This function implements the baseline dropout model for the MNIST
    classification task.
    """

    input_layer = tf.reshape(features, [-1, 28*28])
    
    input_layer = tf.keras.layers.Dropout(0.2).apply(input_layer) 
    
    
    dense_1 = tf.keras.layers.Dense(
        units=params["hidden_units"],
        activation=tf.nn.relu
    ).apply(input_layer)

    #dense_1 = tf.keras.layers.Dropout(params['dropout']).apply(dense_1)
    dense_1 = tf.keras.layers.Dropout(0.5).apply(dense_1) 

    dense_2 = tf.keras.layers.Dense(
        units=params["hidden_units"],
        activation=tf.nn.relu
    ).apply(dense_1)

    #dense_2 = tf.keras.layers.Dropout(params['dropout']).apply(dense_2)
    dense_2 = tf.keras.layers.Dropout(0.2).apply(dense_2)

    dense_out = tf.keras.layers.Dense(
        units=10,
    ).apply(dense_2)

    return dense_out


def dropout_mnist_model_fn(features, labels, mode, params): 
    """
    This function will handle the training, evaluation and prediction procedures.
    """


    if "learning_rate" not in params:
        raise KeyError("No learning rate specified!")

    optimizers = {
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]),
        "adam": tf.train.AdamOptimizer(learning_rate=params["learning_rate"]),
        "rmsprop": tf.train.RMSPropOptimizer(learning_rate=params["learning_rate"])
    }


    logits = create_model(features, params)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = optimizers[params["optimizer"]]
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predictions["classes"])

        # Summary statistic for TensorBoard
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)



