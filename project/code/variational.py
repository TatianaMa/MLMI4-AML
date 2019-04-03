import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

from compression import eliminate_dead_neurons


# ==============================================================================
# Auxiliary functions
# ==============================================================================

def create_gaussian_prior(params):
    prior = tfp.distributions.Normal(loc=params["mu"], scale=tf.exp(-params["sigma"]))
    return prior

def get_reparameterized_log_prob_with_gaussian_prior(prior, inputs, outputs):
    total_var = prior.variance() * tf.reduce_sum(tf.math.square(inputs), axis=1)

    num_units = outputs.get_shape().as_list()[1]
    total_var = tf.reshape(tf.tile(total_var, [num_units]), [-1, num_units])

    return tfp.distributions.Normal(loc=0., scale=total_var).log_prob(outputs)

def kl_normal_normal(mu1, mu2, sigma1, sigma2):
    return tf.math.log(sigma2) - tf.math.log(sigma1) \
        + tf.math.square(sigma1) + tf.math.squared_difference(mu1, mu2) \
        / (2 * tf.math.square(sigma2)) - 0.5

def create_mixture_prior(params):
    prior = tfp.distributions.Mixture(
        cat = tfp.distributions.Categorical(probs=[params["mix_prop"], 1. - params["mix_prop"]]),
        components=[
            tfp.distributions.Normal(loc=0., scale=tf.exp(-params["sigma1"])),
            tfp.distributions.Normal(loc=0., scale=tf.exp(-params["sigma2"])),
        ])
    return prior

def neg_log_prob_with_categorical(logits, labels):
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits)

    return tf.reduce_sum(neg_log_prob)

def neg_log_prob_with_gaussian(predictions, labels, sigma=1.):
    neg_log_prob = tf.losses.mean_squared_error(
        predictions=tf.reshape(predictions, [-1, 1]),
        labels=labels)

    neg_log_prob = neg_log_prob / (2 * sigma**2) + tf.math.log(sigma)

    return neg_log_prob


# ==============================================================================
# Varational estimators
# ==============================================================================


class VarEstimator(snt.AbstractModule):
    """
    Abstract superclass for any variational architecture where some of the layers
    have distributions over the weights
    """

    def __init__(self,
                 prior,
                 prior_params,
                 reparametrisation,
                 name="var_estimator"):

        # Call to super
        super(VarEstimator, self).__init__(name=name)

        # Private fields
        self._layers = []
        self._reparametrisation=reparametrisation

        # Public fields
        self.is_training = True
        self.prior = prior
        self.prior_params = prior_params


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

    def compress(self):
        eliminate_dead_neurons(w_mus=[layer.w_mu.numpy() for layer in self._layers],
                               w_sigmas=[layer.w_sigma.numpy() for layer in self._layers],
                               b_mus=[layer.b_mu.numpy() for layer in self._layers],
                               b_sigmas=[layer.b_sigma.numpy() for layer in self._layers],
                               activations=[tf.nn.relu, tf.nn.relu, lambda x: x])


class VarMushroomRL(VarEstimator):
    """
    Replicates the Q-function estimator from Blundell et al.
    """
    def __init__(self,
                 units,
                 prior,
                 prior_params,
                 reparametrisation,
                 name="var_mushroom_rl"):

        super(VarMushroomRL, self).__init__(prior=prior,
                                            prior_params=prior_params,
                                            reparametrisation=reparametrisation,
                                            name=name)

        self.units = units

    def negative_log_likelihood(self, predictions, labels, sigma=1.):
        return neg_log_prob_with_gaussian(predictions, labels, sigma)

    def _build(self, inputs):

        # Flatten input
        flatten = snt.BatchFlatten()
        flattened = flatten(inputs)

        # First linear layer
        linear_1 = VarLinear(output_size=self.units,
                             reparametrisation=self._reparametrisation,
                             prior=self.prior,
                             prior_params=self.prior_params)

        dense = linear_1(flattened)
        dense = tf.nn.relu(dense)

        # Second linear layer
        linear_2 = VarLinear(output_size=self.units,
                             reparametrisation=self._reparametrisation,
                             prior=self.prior,
                             prior_params=self.prior_params)

        dense = linear_2(dense)
        dense = tf.nn.relu(dense)

        # Final linear layer
        linear_out = VarLinear(output_size=1,
                             reparametrisation=self._reparametrisation,
                               prior=self.prior,
                               prior_params=self.prior_params)

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
                 prior_params,
                 reparametrisation,
                 name="var_regression"):

        super(VarRegression, self).__init__(prior=prior,
                                            prior_params=prior_params,
                                            reparametrisation=reparametrisation,
                                            name=name)

        self.units = units

    def negative_log_likelihood(self, predictions, labels, sigma=1.):
        return neg_log_prob_with_gaussian(predictions, labels, sigma)


    def _build(self, inputs):

        # Flatten input
        flatten = snt.BatchFlatten()
        flattened = flatten(inputs)

        # First linear layer
        linear_1 = VarLinear(output_size=self.units,
                             reparametrisation=self._reparametrisation,
                             prior=self.prior,
                             prior_params=self.prior_params)

        dense = linear_1(flattened)
        dense = tf.nn.relu(dense)

        # Second linear layer
        linear_2 = VarLinear(output_size=self.units,
                             reparametrisation=self._reparametrisation,
                             prior=self.prior,
                             prior_params=self.prior_params)

        dense = linear_2(dense)
        dense = tf.nn.relu(dense)

        # Final linear layer
        linear_out = VarLinear(output_size=1,
                               reparametrisation=self._reparametrisation,
                               prior=self.prior,
                               prior_params=self.prior_params)

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
                 prior_params,
                 reparametrisation,
                 name="var_mnist",
                 **kwargs):

        super(VarMNIST, self).__init__(prior=prior,
                                       prior_params=prior_params,
                                       reparametrisation=reparametrisation,
                                       name=name)

        self.units = units

    def negative_log_likelihood(self, logits, labels):
        return neg_log_prob_with_categorical(logits, labels)

    def _build(self, inputs):

        # Flatten input
        flatten = snt.BatchFlatten()
        flattened = flatten(inputs)

        # First linear layer
        linear_1 = VarLinear(output_size=self.units,
                             reparametrisation=self._reparametrisation,
                             prior=self.prior,
                             prior_params=self.prior_params)

        dense = linear_1(flattened)
        dense = tf.nn.relu(dense)

        # Second linear layer
        linear_2 = VarLinear(output_size=self.units,
                             reparametrisation=self._reparametrisation,
                             prior=self.prior,
                             prior_params=self.prior_params)

        dense = linear_2(dense)
        dense = tf.nn.relu(dense)

        # Final linear layer
        linear_out = VarLinear(output_size=10,
                               reparametrisation=self._reparametrisation,
                               prior=self.prior,
                               prior_params=self.prior_params)

        logits = linear_out(dense)

        self._layers = [linear_1, linear_2, linear_out]

        return logits


class VarLinear(snt.AbstractModule):
    """
    Variational fully-connected layer
    """

    _possible_priors = {
        "gaussian": create_gaussian_prior,
        "mixture": create_mixture_prior
    }

    _possible_reparametrisations = set(['local', 'global'])

    def __init__(self,
                 output_size,
                 prior,
                 prior_params,
                 reparametrisation,
                 use_bias=True,
                 name="var_linear"):

        # Initialise the underlying linear module
        super(VarLinear, self).__init__(name=name)

        self._input_shape = None
        self._use_bias = use_bias

        if reparametrisation not in self._possible_reparametrisations:
            raise Exception("Invalid reparametrisation!")
        self._reparametrisation=reparametrisation

        self.output_size = output_size

        if prior not in self._possible_priors:
            raise Exception("Invalid prior")

        self.prior = prior
        self.prior_params = prior_params


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

        self._prior_dist = self._possible_priors[self.prior](self.prior_params)

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


        if self._reparametrisation == 'global':
            w_dist = tfp.distributions.Normal(loc=self._w_mu,
                                              scale=tf.nn.softplus(self._w_rho))

            w = w_dist.sample()

            # Calculate KL-divergence for later
            self._kl_divergence = tf.reduce_sum(w_dist.log_prob(w) - self._prior_dist.log_prob(w))

            # a = x'W, where W ~ q(W | mu, theta)
            outputs = tf.matmul(inputs, w)

            if self._use_bias:

                b_dist = tfp.distributions.Normal(loc=self._b_mu,
                                                scale=tf.nn.softplus(self._b_rho))

                b = b_dist.sample()
                self._kl_divergence += tf.reduce_sum(b_dist.log_prob(b) \
                                                        - self._prior_dist.log_prob(b))

                # a = x'W, where W ~ q(W | mu, theta), b ~ q(b | mu, theta)
                outputs += b

        elif self._reparametrisation == 'local' and self.prior == 'gaussian':

            hidden_mu = tf.matmul(inputs, self._w_mu) \
                + self._b_mu if self._use_bias else 0

            w_sigma = tf.nn.softplus(self._w_rho)
            b_sigma = tf.nn.softplus(self._b_rho) if self._use_bias else 0

            hidden_sigma = tf.math.sqrt(
                tf.matmul(tf.math.square(inputs),
                          tf.math.square(w_sigma))
                + tf.math.square(b_sigma))

            a_dist = tfp.distributions.Normal(loc=hidden_mu,
                                              scale=hidden_sigma)

            outputs = a_dist.sample()


            # self._kl_divergence = tf.reduce_sum(a_dist.log_prob(outputs) -
            #                                     get_reparameterized_log_prob_with_gaussian_prior(self._prior_dist, inputs, outputs))
            self._kl_divergence = tf.reduce_sum(
                kl_normal_normal(mu1=self._w_mu,
                                 mu2=self._prior_dist.mean(),
                                 sigma1=w_sigma,
                                 sigma2=self._prior_dist.stddev()))
            if self._use_bias:
                self._kl_divergence += tf.reduce_sum(
                    kl_normal_normal(mu1=self._b_mu,
                                     mu2=self._prior_dist.mean(),
                                     sigma1=b_sigma,
                                     sigma2=self._prior_dist.stddev()))

        assert outputs is not None
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

