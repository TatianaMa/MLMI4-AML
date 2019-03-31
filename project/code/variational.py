import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

class VarEstimator(snt.AbstractModule):
    """
    Abstract superclass for any variational architecture where some of the layers
    have distributions over the weights
    """

    def __init__(self,
                 prior,
                 name="var_estimator"):

        # Call to super
        super(VarEstimator, self).__init__(name=name)

        # Private fields
        self._layers = []

        # Public fields
        self.prior = prior


    def negative_log_likelihood(self, logits, labels):
        raise NotImplementedError


    @property
    def kl_divergence(self):
        self._ensure_is_connected()
        return sum([layer.kl_divergence for layer in self._layers])

    @property
    def mu_vector(self):
        self._ensure_is_connected()
        return tf.concat([tf.reshape(layer.w_mu, [-1]) for layer in self._layers] + \
                         [tf.reshape(layer.b_mu, [-1]) for layer in self._layers],
                         axis=0)

    @property
    def sigma_vector(self):
        self._ensure_is_connected()
        return tf.concat([tf.reshape(layer.w_sigma, [-1]) for layer in self._layers] + \
                         [tf.reshape(layer.b_sigma, [-1]) for layer in self._layers],
                         axis=0)

    def prune_below_snr(self, snr, verbose=False):
        self._ensure_is_connected()

        for layer in self._layers:
            layer.prune_below_snr(snr, verbose)

    def sample_posterior(self):
        return tfp.distributions.Normal(loc=self.mu_vector, scale=self.sigma_vector).sample()


class VarMushroomRL(VarEstimator):
    """
    Replicates the Q-function estimator from Blundell et al.
    """
    def __init__(self,
                 units,
                 prior,
                 name="var_mushroom_rl"):

        super(VarMushroomRL, self).__init__(prior=prior,
                                            name=name)

        self.units = units

    def negative_log_likelihood(self, predictions, labels, sigma=1.):
        negative_log_likelihood = tf.losses.mean_squared_error(
            predictions=tf.reshape(predictions, [-1, 1]),
            labels=labels)

        negative_log_likelihood = negative_log_likelihood / (2 * sigma**2) + tf.math.log(sigma)

        return negative_log_likelihood


    def _build(self, inputs):

        # Flatten input
        flatten = snt.BatchFlatten()
        flattened = flatten(inputs)

        # First linear layer
        linear_1 = VarLinear(output_size=self.units,
                             prior=self.prior)

        dense = linear_1(flattened)
        dense = tf.nn.relu(dense)

        # Second linear layer
        linear_2 = VarLinear(output_size=self.units,
                             prior=self.prior)

        dense = linear_2(dense)
        dense = tf.nn.relu(dense)

        # Final linear layer
        linear_out = VarLinear(output_size=1,
                               prior=self.prior)

        logits = linear_out(dense)

        self._layers = [linear_1, linear_2, linear_out]

        return logits


class VarRegression(VarEstimator):
    """
    Replicates the regression task in Blundell et al.
    """

    def __init__(self,
                 units,
                 prior,
                 name="var_regression"):

        super(VarRegression, self).__init__(prior=prior,
                                            name=name)

        self.units = units

    def negative_log_likelihood(self, predictions, labels, sigma=1.):
        negative_log_likelihood = tf.losses.mean_squared_error(
            predictions=tf.reshape(predictions, [-1, 1]),
            labels=labels)

        negative_log_likelihood = negative_log_likelihood / (2 * sigma**2) + tf.math.log(sigma)

        return negative_log_likelihood


    def _build(self, inputs):

        # Flatten input
        flatten = snt.BatchFlatten()
        flattened = flatten(inputs)

        # First linear layer
        linear_1 = VarLinear(output_size=self.units,
                             prior=self.prior)

        dense = linear_1(flattened)
        dense = tf.nn.relu(dense)

        # Second linear layer
        linear_2 = VarLinear(output_size=self.units,
                             prior=self.prior)

        dense = linear_2(dense)
        dense = tf.nn.relu(dense)

        # Final linear layer
        linear_out = VarLinear(output_size=1,
                               prior=self.prior)

        logits = linear_out(dense)

        self._layers = [linear_1, linear_2, linear_out]

        return logits


class VarMNIST(VarEstimator):
    """
    Replicates the MNIST architecture from Blundell et al.
    """
    def __init__(self,
                 units,
                 prior,
                 name="var_mnist"):

        super(VarMNIST, self).__init__(prior=prior,
                                       name=name)

        self.units = units

    def negative_log_likelihood(self, logits, labels):
        negative_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)

        return tf.reduce_sum(negative_log_likelihood)

    def _build(self, inputs):

        # Flatten input
        flatten = snt.BatchFlatten()
        flattened = flatten(inputs)

        # First linear layer
        linear_1 = VarLinear(output_size=self.units,
                             prior=self.prior)

        dense = linear_1(flattened)
        dense = tf.nn.relu(dense)

        # Second linear layer
        linear_2 = VarLinear(output_size=self.units,
                             prior=self.prior)

        dense = linear_2(dense)
        dense = tf.nn.relu(dense)

        # Final linear layer
        linear_out = VarLinear(output_size=10,
                               prior=self.prior)

        logits = linear_out(dense)

        self._layers = [linear_1, linear_2, linear_out]

        return logits


class VarLinear(snt.AbstractModule):
    """
    Variational fully-connected layer
    """

    def __init__(self,
                 output_size,
                 prior,
                 use_bias=True,
                 name="var_linear"):

        # Initialise the underlying linear module
        super(VarLinear, self).__init__(name=name)

        self._input_shape = None
        self._use_bias = use_bias

        self.output_size = output_size
        self.prior = prior


    def prune_below_snr(self, snr, verbose=False):
        self._ensure_is_connected()

        w_snr = 10. * tf.math.log(tf.abs(self._w_mu) / self.w_sigma)
        w_mask = tf.cast(tf.math.greater(w_snr, snr), dtype=tf.float32)

        if verbose:
            num_pruned = tf.reduce_sum(1. - w_mask)

            print("Pruning {} out of {} weights ({:.2f}%) on {}".format(
                int(num_pruned),
                self._num_weights,
                100 * num_pruned / self._num_weights,
                self.module_name))

        self._w_mu.assign(self._w_mu * w_mask)
        self._w_rho.assign(tf.contrib.distributions.softplus_inverse(
            self.w_sigma * w_mask))

        if self._use_bias:
            b_snr = 10. * tf.math.log(tf.abs(self._b_mu) / self.b_sigma)
            b_mask = tf.cast(tf.math.greater(b_snr, snr), dtype=tf.float32)

            if verbose:
                num_pruned = tf.reduce_sum(1. - b_mask)

                print("Pruning {} out of {} biases ({:.2f}%) on {}".format(
                    int(num_pruned),
                    self._num_biases,
                    100 * num_pruned / self._num_biases,
                    self.module_name))

            self._b_mu.assign(self._b_mu * b_mask)
            self._b_rho.assign(tf.contrib.distributions.softplus_inverse(
                self.b_sigma * b_mask))

    def _build(self, inputs):

        # ======================================================================
        # Ensure the input has the correct size
        # ======================================================================
        input_shape = tuple(inputs.get_shape().as_list())

        if len(input_shape) != 2:
            raise base.IncompatibleShapeError(
                "{}: rank of shape must be 2 not: {}".format(
                    self.scope_name, len(input_shape)))

        if input_shape[1] is None:
            raise base.IncompatibleShapeError(
                "{}: Input size must be specified at module build time".format(
                    self.scope_name))

        if self._input_shape is not None and input_shape[1] != self._input_shape[1]:
            raise base.IncompatibleShapeError(
                "{}: Input shape must be [batch_size, {}] not: [batch_size, {}]"
                .format(self.scope_name, self._input_shape[1], input_shape[1]))

        # ======================================================================
        # Initialise parameters
        # ======================================================================
        self._input_shape = input_shape
        dtype = inputs.dtype

        mu_init = tf.initializers.glorot_uniform()
        rho_init = tf.initializers.constant(-3)

        weight_shape = (self._input_shape[1], self.output_size)

        self._num_weights = weight_shape[0] * weight_shape[1]

        # Weight parameters
        self._w_mu = tf.get_variable("w_mu",
                                     shape=weight_shape,
                                     dtype=dtype,
                                     initializer=mu_init)
        self._w_rho = tf.get_variable("w_rho",
                                      shape=weight_shape,
                                      dtype=dtype,
                                      initializer=rho_init)

        w_dist = tfp.distributions.Normal(loc=self._w_mu,
                                          scale=tf.nn.softplus(self._w_rho))

        w = w_dist.sample()

        # Calculate KL-divergence for later
        self._kl_divergence = tf.reduce_sum(w_dist.log_prob(w) - self.prior.log_prob(w))

        # a = x'W, where W ~ q(W | mu, theta)
        outputs = tf.matmul(inputs, w)

        if self._use_bias:
            bias_shape = (self.output_size,)

            self._num_biases = self.output_size

            self._b_mu = tf.get_variable("b_mu",
                                         shape=bias_shape,
                                         dtype=dtype,
                                         initializer=mu_init)
            self._b_rho = tf.get_variable("b_rho",
                                          shape=bias_shape,
                                          dtype=dtype,
                                          initializer=rho_init)

            b_dist = tfp.distributions.Normal(loc=self._b_mu,
                                              scale=tf.nn.softplus(self._b_rho))

            b = b_dist.sample()
            self._kl_divergence += tf.reduce_sum(b_dist.log_prob(b) - self.prior.log_prob(b))


            # a = x'W, where W ~ q(W | mu, theta), b ~ q(b | mu, theta)
            outputs += b

        return outputs


    @property
    def kl_divergence(self):
        self._ensure_is_connected()
        return self._kl_divergence

    @property
    def w_mu(self):
        self._ensure_is_connected()
        return self._w_mu

    @property
    def w_rho(self):
        self._ensure_is_connected()
        return self._w_rho

    @property
    def w_sigma(self):
        self._ensure_is_connected()
        return tf.nn.softplus(self._w_rho)

    @property
    def b_mu(self):
        self._ensure_is_connected()
        return self._b_mu

    @property
    def b_rho(self):
        self._ensure_is_connected()
        return self._b_rho

    @property
    def b_sigma(self):
        self._ensure_is_connected()
        return tf.nn.softplus(self._b_rho)

