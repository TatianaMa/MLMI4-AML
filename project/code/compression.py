import numpy as np
import tensorflow as tf

def snr(mus, sigmas):
    if mus.shape != sigmas.shape:
        raise base.IncompatibleShapeError(
            "Size of the mus {} has to match the size of the sigmas {}".format(mus.shape,
                                                                               sigmas.shape)
        )

    return 10. * (np.log(np.abs(mus)) - np.log(sigmas))

def eliminate_dead_neurons(w_mus, w_sigmas, b_mus, b_sigmas, activations):
    """
    weights - list of weights in the neural network
    biases - list of biases
    activation - list of activation functions for each layer
    """

    num_layers = len(w_mus)

    # Backward pass
    for i in range(num_layers - 1, -1, -1):

        w_mu = w_mus[i]
        w_sigma = w_sigmas[i]

        num_rows = w_mu.shape[0]

        # Check for a zero-column
        # Only weights set by pruning will have 0 variance
        keep_indices = [j for j in range(num_rows) if not np.all(w_sigma[j, :] == 0)]

        print(len(keep_indices))

        # Remove rows on current layer
        w_mus[i] = w_mu[keep_indices, :]
        w_sigmas[i] = w_sigma[keep_indices, :]

        # Remove columns on previous layer and unused biases
        if i > 0:
            w_mus[i - 1] = w_mus[i - 1][:, keep_indices]
            w_sigmas[i - 1] = w_sigmas[i - 1][:, keep_indices]

            b_mus[i - 1] = b_mus[i - 1][keep_indices]
            b_sigmas[i - 1] = b_sigmas[i - 1][keep_indices]
        else:
            kept_input_indices = keep_indices

        print("Shapes: {}, {}".format(w_mus[i].shape, b_mus[i].shape))

    print(np.sum(w_mus[2] != 0., axis=1))

    # Forward pass
    for i in range(num_layers):

        w_mu = w_mus[i]
        w_sigma = w_sigmas[i]

        num_cols = w_mu.shape[1]

        # Check for a zero-row
        # Only weights set by pruning will have 0 variance
        keep_indices = [j for j in range(num_cols) if not np.all(w_sigma[:, j] == 0)]

        print(len(keep_indices))

        # Remove columns on current layer
        w_mus[i] = w_mu[:, keep_indices]
        w_sigmas[i] = w_sigma[:, keep_indices]

        # Remove rows on next layer and absorb dead neurons into biases
        if i > 0:
            w_mus[i - 1] = w_mus[i - 1][keep_indices, :]
            w_sigmas[i - 1] = w_sigmas[i - 1][keep_indices, :]

            b_mus[i - 1] = b_mus[i - 1][keep_indices]
            b_sigmas[i - 1] = b_sigmas[i - 1][keep_indices]
        else:
            kept_input_indices = keep_indices

        print("Shapes: {}, {}".format(w_mus[i].shape, b_mus[i].shape))


    return w_mus, w_sigmas, b_mus, b_sigmas


# w1 = np.eye(6)
# w1[1,1] = 0
# w1[5,5] = 0

# w2 = np.eye(6)
# w2[2,2] = 0
# w2[3,3] = 0

# b1 = np.ones((6,))
# b2 = np.zeros((6,))


# w, b, i = eliminate_dead_neurons([w1, w2], [b1, b2], [])

# print(w)
# print(b)
# print(i)
