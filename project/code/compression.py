import numpy as np
import tensorflow as tf

def snr(mus, sigmas):
    if mus.shape != sigmas.shape:
        raise base.IncompatibleShapeError(
            "Size of the mus {} has to match the size of the sigmas {}".format(mus.shape,
                                                                               sigmas.shape)
        )

    return 10. * (np.log10(np.abs(mus)) - np.log10(sigmas))

def eliminate_dead_neurons(w_mus, w_sigmas, b_mus, b_sigmas, activations):
    """
    weights - list of weights in the neural network
    biases - list of biases
    activation - list of activation functions for each layer
    """

    old_num_zeros = 0

    for i in range(len(w_mus)):
        old_num_zeros += np.sum((w_mus[i] == 0).astype(np.float32) * (w_sigmas[i] == 0).astype(np.float32))
        old_num_zeros += np.sum((b_mus[i] == 0).astype(np.float32) * (b_sigmas[i] == 0).astype(np.float32))

    num_layers = len(w_mus)

    backward_non_zeros_removed = 0

    # Backward pass
    for i in range(num_layers - 1, -1, -1):

        w_mu = w_mus[i]
        w_sigma = w_sigmas[i]

        num_rows = w_mu.shape[0]

        # Check for a zero-column
        # Only weights set by pruning will have 0 variance
        keep_indices = [j for j in range(num_rows) if not np.all(w_sigma[j, :] == 0)]
        drop_indices = [j for j in range(num_rows) if np.all(w_sigma[j, :] == 0)]

        #print(len(keep_indices))

        # Remove rows on current layer
        w_mus[i] = w_mu[keep_indices, :]
        w_sigmas[i] = w_sigma[keep_indices, :]

        # Remove columns on previous layer and unused biases
        if i > 0:
            backward_non_zeros_removed += \
                np.sum((w_mus[i - 1][:, drop_indices] != 0).astype(np.float32) + (w_sigmas[i - 1][:, drop_indices] != 0).astype(np.float32))
            backward_non_zeros_removed += \
                np.sum((b_mus[i - 1][drop_indices] != 0).astype(np.float32) + (b_sigmas[i - 1][drop_indices] != 0).astype(np.float32))

            w_mus[i - 1] = w_mus[i - 1][:, keep_indices]
            w_sigmas[i - 1] = w_sigmas[i - 1][:, keep_indices]

            b_mus[i - 1] = b_mus[i - 1][keep_indices]
            b_sigmas[i - 1] = b_sigmas[i - 1][keep_indices]
        else:
            kept_input_indices = keep_indices

        #print("Shapes: {}, {}".format(w_mus[i].shape, b_mus[i].shape))

    print("Backwards pass removed {} non-zero elements.".format(backward_non_zeros_removed))
    #print(np.sum(w_mus[2] != 0., axis=1))

    forward_non_zeros_removed = 0
    # Forward pass
    for i in range(num_layers):

        w_mu = w_mus[i]
        w_sigma = w_sigmas[i]

        num_cols = w_mu.shape[1]

        # Check for a zero-row
        # Only weights set by pruning will have 0 variance
        keep_indices = [j for j in range(num_cols) if not np.all(w_sigma[:, j] == 0)]
        drop_indices = [j for j in range(num_cols) if np.all(w_sigma[:, j] == 0)]

        #print(len(keep_indices))

        # Remove columns on current layer
        w_mus[i] = w_mu[:, keep_indices]
        w_sigmas[i] = w_sigma[:, keep_indices]

        # Remove rows on next layer and absorb dead neurons into biases
        if i < num_layers - 1:

            forward_non_zeros_removed += \
                np.sum((w_mus[i + 1][drop_indices, :] != 0).astype(np.float32) + (w_sigmas[i + 1][drop_indices, :] != 0).astype(np.float32))
            forward_non_zeros_removed += \
                np.sum((b_mus[i][drop_indices] != 0).astype(np.float32) + (b_sigmas[i][drop_indices] != 0).astype(np.float32))



            w_mus[i + 1] = w_mus[i + 1][keep_indices, :]
            w_sigmas[i + 1] = w_sigmas[i + 1][keep_indices, :]

            b_mus[i] = b_mus[i][keep_indices]
            b_sigmas[i] = b_sigmas[i][keep_indices]

        print("Shapes: {}, {}".format(w_mus[i].shape, b_mus[i].shape))

    print("Forward pass removed {} non-zero elements.".format(forward_non_zeros_removed))
    num_zeros = 0

    for i in range(len(w_mus)):
        num_zeros += np.sum((w_mus[i] == 0).astype(np.float32) * (w_sigmas[i] == 0).astype(np.float32))
        num_zeros += np.sum((b_mus[i] == 0).astype(np.float32) * (b_sigmas[i] == 0).astype(np.float32))


    print("Old # 0: {}, New # 0: {}, Ratio: {:.2f}%".format(old_num_zeros, num_zeros, 100 * float(num_zeros)/ old_num_zeros))

    return kept_input_indices, w_mus, w_sigmas, b_mus, b_sigmas


def densify_weight_matrix(w):
     pass
