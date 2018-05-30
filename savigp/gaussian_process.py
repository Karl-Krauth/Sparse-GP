"""
This module implements the GaussianProcess base class.

Details about the model can be found at the NIPS paper below along with its supplemental material:
- http://ebonilla.github.io/papers/dezfouli-bonilla-nips-2015.pdf
- http://ebonilla.github.io/papers/dezfouli-bonilla-nips-2015-supplemental.pdf
Many of the gradients described by the papers need to go through some transformation when
implemented, this is described in the README. TODO(karl): Write the README.

For the sake of code cleanliness and readability we renamed many of the single letter variables
in the the paper to longer variables. The mappings from paper names to program names are:
- N -> num_data_points
- M -> num_inducing
- D -> input_dim
- K -> num_components
- Q -> num_latent
- S -> num_samples
- x_n -> train_input
- X -> train_inputs, input_partitions
- y_n -> train_output
- Y -> train_outputs, output_partitions
- Z -> inducing_locations
- K -> kernel_matrix
- \kappa(X, Z_j) -> data_inducing_kernel
- A -> kernel_products
- \widetilde{\K} -> diag_conditional_covars
- \mathcal{N} -> normal_samples
- b -> sample_means
- \Sigma -> sample_vars
- f^{(i)} -> samples
- m -> gaussian_mixture.means
- S -> gaussian_mixture.covars
- \pi -> gaussian_mixture.weights
- \theta -> hyper_params
- \delta(h)_p -> grad_h_over_p
"""
import multiprocessing.pool
import threading
import warnings

import GPy
from GPy.util.linalg import mdot
import numpy as np
import scipy.linalg
import scipy.misc
import sklearn.cluster
import theano
from theano import tensor
from theano.sandbox import rng_mrg

import util

# A list of possible sets of parameters ordered according to optimization ordering.
PARAMETER_SETS = ['hyp', 'mog', 'hyp', 'll', 'inducing']


class GaussianProcess(object):
    """
    The base class for scalable automated variational inference for Gaussian process models.

    The class represents a model whose parameters can be optimized. An optimization run
    will usually involve the following sequence of calls:
    - set_optimization_method to set the subset of parameters that get optimized.
    - get_params to inspect the state of the model.
    - objective_function_gradients and set_params to update the model accordingly.
    - objective_function and get_gaussian_mixture_params to check for convergence.

    The subsets of parameters that get optimized according to the optimization_method string are:
    -'mog' -> {gaussian_mixture.means, gaussian_mixture.covars, gaussian_mixture.weights}
    -'hyper' -> {kernel.param_array}
    -'ll' -> {likelihood.params}
    -'inducing' -> {inducing_locations}

    Parameters
    ----------
    train_inputs : ndarray
        A matrix containing the input training data. Dimensions: num_data_points * input_dim.
    train_outputs : ndarray
        A matrix containing the output training data. Dimensions: num_data_points * num_latent.
    num_inducing : int
        The number of inducing points to use in training.
    num_components : int
        The number of components for the mixture of Gaussians posterior.
    num_samples : int
        The number of samples used to approximate gradients and objective functions.
    likelihood : subclass of Likelihood
        An object representing the likelihood function.
    kernels : list of kernels
        A list containing a kernel for each latent process.
    latent_noise : float
        The amount of latent noise that will be added to each kernel.
    exact_ell : boolean
        Whether to use the exact log likelihood provided by the likelihood method or approximation
        via sampling.
    inducing_on_inputs: boolean
        Whether to put inducing points randomly on training data. If False, inducing points will be
        determined using clustering.
    num_threads : int
        The number of threads to use for calculating the expected log likelihood and its gradients.
    partition_size : int
        How large each partition of the training data will be when calculating expected log
        likelihood.
    """
    def __init__(self,
                 train_inputs,
                 train_outputs,
                 num_inducing,
                 num_components,
                 num_samples,
                 kernels,
                 likelihood,
                 latent_noise=0,
                 exact_ell=False,
                 inducing_on_inputs=False,
                 num_threads=1,
                 partition_size=3000,
                 GP_mean=None,
                 init_var=None
                 ):
        train_inputs = train_inputs.astype(np.float32)
        train_outputs = train_outputs.astype(np.float32)

        # mean for Gaussian Processes
        self.num_latent = len(kernels)
        if GP_mean is None:
            self.GP_mean = np.zeros(self.num_latent)
        else:
            self.GP_mean = GP_mean

        # Initialize variables to keep track of various model dimensions.
        self.num_components = num_components
        self.num_inducing = num_inducing
        self.num_samples = num_samples
        self.num_hyper_params = kernels[0].gradient.shape[0]
        self.num_likelihood_params = likelihood.get_num_params()
        self.num_data_points = train_inputs.shape[0]
        self.input_dim = train_inputs.shape[1]
        self.partition_size = partition_size

        # Initialize training data and functions.
        self.input_partitions = self._partition_data(self.partition_size, train_inputs)
        self.output_partitions = self._partition_data(self.partition_size, train_outputs)
        self.train_index_start = 0
        self.train_len = len(self.input_partitions)
        self.kernels = kernels
        self.likelihood = likelihood

        # Initialize model configuration information.
        self.latent_noise = latent_noise
        self.num_threads = num_threads
        self.is_exact_ell = exact_ell
        self.optimization_method = 'mog'

        # Initialize the parameters to optimize.
        self.inducing_locations, initial_mean = (
            self._initialize_inducing_points(train_inputs, train_outputs, inducing_on_inputs))
        self.gaussian_mixture = self._get_gaussian_mixture(initial_mean, init_var)
        self.hyper_params = np.array([self.kernels[i].param_array.copy()
                                      for i in xrange(self.num_latent)], dtype=np.float32)

        # Initialize the interim variables used to calculate parameters.
        self.cached_ell = None
        self.cached_entropy = None
        self.cached_cross = None
        self.kernel_matrix = util.PosDefMatrix(self.num_latent, self.num_inducing)
        self.curr_log_likelihood_gradients = None

        # Update the model.
        self._update_latent_kernel()
        self._update_log_likelihood()

    def set_optimization_method(self, optimization_method):
        """
        Set which subset of parameters will be retrieved and updated by get_params and set_params
        respectively.

        Parameters
        ----------
        optimization_method : str
            The subset of parameters to be considered. Possible values include: 'mog', 'hyp', 'll',
            and 'inducing'.
        """
        if self.optimization_method == optimization_method:
            return

        self.optimization_method = optimization_method
        self.cached_ell = None
        self.cached_cross = None
        self.cached_entropy = None
        self._update_log_likelihood()

    def set_train_partitions(self, train_index_start, train_len=1):
        assert train_index_start >= 0
        assert train_index_start + train_len <= self.get_num_partitions()
        assert self.get_num_partitions() % train_len == 0

        self.train_index_start = train_index_start
        self.train_len = train_len

    def get_num_partitions(self):
        return len(self.input_partitions)

    def shuffle_data(self):
        partition_size = self.input_partitions[0].shape[0]

        train_inputs = np.concatenate(self.input_partitions)
        train_outputs = np.concatenate(self.output_partitions)
        np.random.permutation(train_inputs)
        np.random.permutation(train_outputs)

        self.input_partitions = self._partition_data(partition_size, train_inputs)
        self.output_partitions = self._partition_data(partition_size, train_outputs)

    def set_params(self, new_params):
        """
        Update the subset of the model parameters that are currently under consideration.

        Parameters
        ----------
        new_params : ndarray
            An array of values to set the model parameters to. Dimension varies according to the
            current optimization method.
        """
        new_params = new_params.astype(np.float32)
        if self.optimization_method == 'mog':
            self.gaussian_mixture.set_params(new_params)
        elif self.optimization_method == 'hyp':
            self.hyper_params = np.exp(new_params[:].reshape([self.num_latent,
                                                              self.num_hyper_params]))
            for i in xrange(self.num_latent):
                self.kernels[i].param_array[:] = self.hyper_params[i].copy()
            self._update_latent_kernel()
        elif self.optimization_method == 'll':
            self.likelihood.set_params(new_params)
        elif self.optimization_method == 'inducing':
            self.inducing_locations = new_params.reshape([
                self.num_latent, self.num_inducing, self.input_dim])
            self.kernel_matrix.set_outdated()

        self._update_log_likelihood()

    def get_params(self):
        """
        Get the subset of the model parameters that are currently under consideration.

        Returns
        -------
        ndarray
            An array of the model parameters whose dimension varies according to the current
            optimization method.
        """
        if self.optimization_method == 'mog':
            return self.gaussian_mixture.get_params()
        elif self.optimization_method == 'hyp':
            return np.log(self.hyper_params.flatten())
        elif self.optimization_method == 'll':
            return self.likelihood.get_params()
        elif self.optimization_method == 'inducing':
            return self.inducing_locations.flatten()

    def get_gaussian_mixture_params(self):
        """
        Get the current parameters of the mixture of gaussian.

        Returns
        -------
        means : ndarray
            The current mixture of Gaussians means. Dimensions: num_components * num_inducing.
        covars : ndarray
            The current mixture of Gaussians covariances. Dimensions vary according to whether the
            model uses a full or diagonal mixture of Gaussians.
        """
        return self.gaussian_mixture.get_means_and_covars()

    def overall_objective_function(self):
        ell = np.float32(0.0)
        for input_partition, output_partition in zip(self.input_partitions, self.output_partitions):
            data_inducing_kernel, kernel_products, diag_conditional_covars = (
                self._get_interim_matrices(input_partition))

            for i in xrange(self.num_components):
                # Pre-compute values relevant to calculating the ell.
                partition_size = input_partition.shape[0]
                normal_samples, sample_means, sample_vars, samples = (
                    self._get_samples_partition(i, partition_size, kernel_products, diag_conditional_covars))
                conditional_ll, _ = self.likelihood.ll_F_Y(samples, output_partition)
                conditional_ll = conditional_ll.astype(np.float32)

                # Now compute ell for this component.
                ell += self._calculate_ell(i, output_partition, conditional_ll,
                                           sample_means, sample_vars)

        cross = self._calculate_cross(self._grad_cross_over_weights())
        return -((self._calculate_entropy() + cross) + ell)

    def objective_function(self):
        """
        Get the current negative log likelihood value.

        Returns
        -------
        float
            The current negative log likelihood value.
        """
        return -(self.cached_entropy + self.cached_cross + self.cached_ell)

    def objective_function_gradients(self):
        """Gets the current negative log likelihood gradients."""
        return -self.curr_log_likelihood_gradients.copy()

    def predict(self, test_inputs, test_outputs=None):
        """
        Make predictions on test inputs and computes the negative log predictive density for the
        test outputs if they are given.

        Parameters
        ----------
        test_inputs : ndarray
            Dimensions: num_test * input_dim.
        test_outputs : ndarray
            Dimensions: num_test * output_dim.

        Returns
        -------
        predicted_values : ndarray
            Predicted values for each test point. Dimensions : num_test * output_dim.
        predicted_variance : ndarray
            Predicted variance of the values. Dimensions : num_test * output_dim
        nlpd : ndarray
            The negative log predictive density for each test point or None if test outputs aren't
            provided. Dimensions: num_test
        """
        # TODO(karl): Make this nicer.
        num_partitions = (self.num_data_points + self.partition_size - 1) / self.partition_size
        test_inputs = test_inputs.astype(np.float32)
        input_partitions = np.array_split(test_inputs, num_partitions)
        if test_outputs is not None:
            test_outputs = test_outputs.astype(np.float32)
            output_partitions = np.array_split(test_outputs, num_partitions)
        else:
            output_partitions = [None] * len(input_partitions)

        mu, var, nlpd = self._predict_partition(input_partitions[0], output_partitions[0])
        for input_partition, output_partition in zip(input_partitions[1:], output_partitions[1:]):
            temp_mu, temp_var, temp_nlpd = self._predict_partition(input_partition,
                                                                   output_partition)
            mu = np.concatenate([mu, temp_mu])
            var = np.concatenate([var, temp_var])
            if nlpd is not None:
                nlpd = np.concatenate([nlpd, temp_nlpd])

        predicted_values = np.average(mu, axis=1, weights=self.gaussian_mixture.weights)
        predicted_variance = (np.average(mu ** 2, axis=1, weights=self.gaussian_mixture.weights) +
                              np.average(var, axis=1, weights=self.gaussian_mixture.weights) -
                              predicted_values ** 2)

        return predicted_values, predicted_variance, nlpd

    def get_samples_posterior(self, test_inputs, num_samples=None):
        """
        Get samples from the posterior
        :param test_inputs: 
        :param num_samples 
        :return all_samples: (Ns, N, Q)-array with Ns samples for all N datapoints and Q latent functions
        """
        if num_samples is None:
            num_samples = self.num_samples

        num_partitions = (self.num_data_points + self.partition_size - 1) / self.partition_size
        test_inputs = test_inputs.astype(np.float32)
        input_partitions = np.array_split(test_inputs, num_partitions)
        N = test_inputs.shape[0]

        all_samples = np.empty([num_samples, N, self.num_latent],
                               dtype=np.float32)
        all_means = np.empty([self.num_latent, N, self.num_components],
                             dtype=np.float32)
        all_vars = np.empty([self.num_latent, N, self.num_components],
                             dtype=np.float32)
        normal_samples = np.random.normal(0.0, 1.0, [num_samples, self.num_latent, N])

        # Compute all means and variances of posteriors for all latent functions
        ptr_low = 0
        for input_partition in input_partitions:
            partition_size = input_partition.shape[0]
            ptr_high = ptr_low + partition_size
            data_inducing_kernel, kernel_products, diag_conditional_covars = (
                self._get_interim_matrices(input_partition))

            for k in xrange(self.num_components):

                for j in xrange(self.num_latent):
                    kern_dot_covar_dot_kern = self.gaussian_mixture.a_dot_covar_dot_a(
                        kernel_products[j], k, j)
                    all_means[j, ptr_low:ptr_high, k], all_vars[j, ptr_low:ptr_high, k] = (
                        self._theano_get_means_vars_f_partition(kernel_products[j],
                                                                 diag_conditional_covars[j],
                                                                 kern_dot_covar_dot_kern,
                                                                 self.gaussian_mixture.means[k, j]))
            ptr_low = ptr_high

        cat_sample = np.transpose(np.nonzero(np.random.multinomial(n=1,
                                                                    pvals=self.gaussian_mixture.weights,
                                                                    size=N*self.num_latent)))[:,1]
        idx_mixture = np.reshape(cat_sample, (N,self.num_latent))

        # for every latent function, sample from its mixture distribution
        for j in xrange(self.num_latent):
            component = np.squeeze(idx_mixture[:, j])
            idx_all = xrange(N)
            all_samples[:, :, j] = normal_samples[:, j, :] * np.sqrt(all_vars[j, idx_all, component]) + \
                all_means[j, idx_all, component]

        return all_samples


    def _compile_get_means_vars_f_partition():
        kernel_products = tensor.matrix('kernel_products')
        diag_conditional_covars = tensor.vector('diag_conditional_covars')
        kern_dot_covars_dot_kern = tensor.vector('kern_dot_covars_dot_kern')
        gaussian_mixture_means = tensor.vector('gaussian_mixture_means')
        partition_size = kernel_products.shape[0]

        sample_means = tensor.dot(kernel_products, gaussian_mixture_means.T)
        sample_vars = diag_conditional_covars + kern_dot_covars_dot_kern

        return theano.function([kernel_products, diag_conditional_covars, kern_dot_covars_dot_kern,
                                gaussian_mixture_means], [sample_means, sample_vars])

    _theano_get_means_vars_f_partition = _compile_get_means_vars_f_partition()






    @staticmethod
    def _partition_data(partition_size, train_data):
        num_partitions = ((train_data.shape[0] + partition_size - 1) / partition_size)
        return np.array_split(train_data, num_partitions)

    def _get_gaussian_mixture(self, initial_mean):
        """Get the mixture of Gaussians used for representing the posterior distribution."""
        raise NotImplementedError

    def _initialize_inducing_points(self, train_inputs, train_outputs, inducing_on_inputs):
        """
        Initialize the position of inducing points and the initial posterior distribution means.

        Parameters
        ----------
        train_inputs : ndarray
            Input data. Dimensions: num_train * input_dim.
        train_outputs : ndarray
            Output data. Dimensions: num_train * output_dim.
        inducing_on_inputs : bool
            If True, initializes the inducing points on the input data otherwise, inducing points
            are initialized using clustering.

        Returns
        -------
        inducing_locations : ndarray
            An array of inducing point locations. Dimensions: num_latent * num_inducing * input_dim.
        initial_mean : ndarray
            Initial value for the mean of the posterior distribution.
            Dimensions: num_inducing * num_latent.
        """
        inducing_locations = np.zeros([self.num_latent, self.num_inducing, self.input_dim],
                                      dtype=np.float32)
        initial_mean = np.empty([self.num_latent, self.num_inducing], dtype=np.float32)

        if inducing_on_inputs or self.num_inducing == self.num_data_points:
            # Initialize inducing points on training data.
            for i in xrange(self.num_latent):
                inducing_index = np.random.permutation(self.num_data_points)[:self.num_inducing]
                inducing_locations[i] = train_inputs[inducing_index]
            for i in xrange(self.num_inducing):
                initial_mean[:, i] = self.likelihood.map_Y_to_f(train_outputs[inducing_index[i]])
        else:
            # Initialize inducing points using clustering.
            mini_batch = sklearn.cluster.MiniBatchKMeans(self.num_inducing)
            with warnings.catch_warnings():
                # Squash deprecation warning in some older versions of scikit.
                warnings.simplefilter("ignore")
                cluster_indices = mini_batch.fit_predict(train_inputs)

            for i in xrange(self.num_latent):
                inducing_locations[i] = mini_batch.cluster_centers_
            for i in xrange(self.num_inducing):
                data_indices, = np.where(cluster_indices == i)
                if data_indices.shape[0] == 0:
                    # No points in this cluster so set the mean across all data points.
                    initial_mean[:, i] = self.likelihood.map_Y_to_f(train_outputs)
                else:
                    initial_mean[:, i] = self.likelihood.map_Y_to_f(train_outputs[data_indices])

        return inducing_locations, initial_mean

    def _update_log_likelihood(self):
        """
        Updates objective function and its gradients under current configuration and stores them in
        the corresponding variables for future uses.
        """
        self.kernel_matrix.update(self.kernels, self.inducing_locations)
        num_batches = len(self.input_partitions) / self.train_len
        # Update the entropy and cross entropy components.
        if self.optimization_method != 'll' or self.cached_entropy is None:
            self.cached_entropy = self._calculate_entropy() / num_batches
        if self.optimization_method != 'll' or self.cached_cross is None:
            grad_cross_over_weights = self._grad_cross_over_weights()
            self.cached_cross = self._calculate_cross(grad_cross_over_weights) / num_batches

        # Update the objective gradients and the ell component.
        if self.optimization_method == 'mog':
            self.cached_ell, grad_ell_over_means, grad_ell_over_covars, grad_ell_over_weights = (
                self._apply_over_data(self._gaussian_mixture_ell))
            means_grad = (
                (self._grad_entropy_over_means() + self._grad_cross_over_means()) / num_batches +
                grad_ell_over_means)
            covars_grad = (
                (self._grad_entropy_over_covars() + self._grad_cross_over_covars()) / num_batches +
                self.gaussian_mixture.transform_covars_grad(grad_ell_over_covars))
            weights_grad = (
                (self._grad_entropy_over_weights() + grad_cross_over_weights) / num_batches +
                grad_ell_over_weights)
            self.curr_log_likelihood_gradients = np.hstack([
                means_grad.flatten(), covars_grad,
                self.gaussian_mixture.transform_weights_grad(weights_grad)])
        elif self.optimization_method == 'hyp':
            self.cached_ell, grad_ell_over_hyper_params = self._apply_over_data(
                self._hyper_params_ell)
            for i in xrange(self.num_latent):
                self.hyper_params[i] = self.kernels[i].param_array.copy()
            grad_hyper = (
                self._grad_cross_over_hyper_params() / num_batches + grad_ell_over_hyper_params)
            self.curr_log_likelihood_gradients = grad_hyper.flatten() * self.hyper_params.flatten()
        elif self.optimization_method == 'll':
            self.cached_ell, grad_ell_over_likelihood_params = self._apply_over_data(
                self._likelihood_params_ell)
            self.curr_log_likelihood_gradients = grad_ell_over_likelihood_params
        elif self.optimization_method == 'inducing':
            self.cached_ell, grad_ell_over_inducing = self._apply_over_data(self._inducing_ell)
            grad_inducing = self._grad_cross_over_inducing() / num_batches + grad_ell_over_inducing
            self.curr_log_likelihood_gradients = grad_inducing.flatten()

    def _update_latent_kernel(self):
        """Update kernels by adding latent noise to all of them."""
        self.kernels_latent = [
            self.kernels[i] + GPy.kern.White(self.input_dim, variance=self.latent_noise)
            for i in xrange(self.num_latent)]
        self.kernel_matrix.set_outdated()

    def _apply_over_data(self, func):
        """
        Take a function, apply it concurrently over the data partitions, and return the sum
        of the result of each function.

        Parameters
        ----------
        func : callable
            A function that takes an input partition and an output partition and returns a tuple of
            elements that support the add operator.

        Returns
        -------
        tuple
            The element-wise sum of the return value of all calls to func.
        """
        lock = threading.Lock()
        final_result = []

        def func_wrapper((input_partition, output_partition)):
            try:
                import time
                start = time.time()
                result = func(input_partition, output_partition)
                # print time.time() - start
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
            with lock:
                if not final_result:
                    final_result.append(result)
                else:
                    final_result[0] = map(sum, zip(final_result[0], result))

        # thread_pool = multiprocessing.pool.ThreadPool(processes=self.num_threads)
        train_index_end = self.train_index_start + self.train_len
        func_args = zip(self.input_partitions[self.train_index_start:train_index_end],
                        self.output_partitions[self.train_index_start:train_index_end])
        # thread_pool.map(func_wrapper, func_args)
        map(func_wrapper, func_args)
        # thread_pool.close()

        return final_result[0]

    def _gaussian_mixture_ell(self, input_partition, output_partition):
        """
        Calculate the expected log likelihood alongside its gradients with respect to mixture of
        Gaussian parameters.

        Parameters
        ----------
        input_partition : ndarray
            The input data. Dimensions: partition_size * input_dim.
        output_partition : ndarray
            The output data. Dimensions: partition_size * output_dim.

        Returns
        -------
        ell : float
            The value of the expected log likelihood over the given data.
        means_grad : ndarray
            The gradient of the ell with respect to the means of the mixture of Gaussians.
        covars_grad : ndarray
            The gradient of the ell with respect to the covariance of the mixture of Gaussians.
        weights_grad : ndarray
            The gradient of the ell with respect to the weights of the mixture of Gaussians.
        """
        ell = 0
        means_grad = np.empty([self.num_components, self.num_latent, self.num_inducing],
                              dtype=np.float32)
        covars_grad = np.empty([self.num_components, self.num_latent] +
                               self.gaussian_mixture.get_covar_shape(), dtype=np.float32)
        weights_grad = np.empty(self.num_components, dtype=np.float32)
        data_inducing_kernel, kernel_products, diag_conditional_covars = (
            self._get_interim_matrices(input_partition))
        for i in xrange(self.num_components):
            # Pre-compute values relevant to calculating the gradients and ell.
            partition_size = input_partition.shape[0]
            normal_samples, sample_means, sample_vars, samples = (
                self._get_samples_partition(i, partition_size, kernel_products, diag_conditional_covars))
            conditional_ll, _ = self.likelihood.ll_F_Y(samples, output_partition)
            conditional_ll = conditional_ll.astype(np.float32)

            # Now compute gradients and ell for this component.
            ell += self._calculate_ell(
                i, output_partition, conditional_ll, sample_means, sample_vars)
            means_grad[i] = self._grad_ell_over_means(
                i, conditional_ll, data_inducing_kernel, sample_vars, normal_samples)
            covars_grad[i] = self._grad_ell_over_covars(
                i, conditional_ll, kernel_products, sample_vars, normal_samples)
            weights_grad[i] = conditional_ll.sum() / self.num_samples

        return ell, means_grad, covars_grad, weights_grad

    def _hyper_params_ell(self, input_partition, output_partition):
        """
        Calculate the expected log likelihood alongside its gradients with respect to
        the kernel hyper-parameters.

        Parameters
        ----------
        input_partition : ndarray
            The input data. Dimensions: partition_size * input_dim.
        output_partition : ndarray
            The output data. Dimensions: partition_size * output_dim.

        Returns
        -------
        ell : float
            The value of the expected log likelihood over the given data.
        hyper_params_grad : ndarray
            The gradient of the ell with respect to the kernel hyper-parameters.
        """
        ell = 0
        hyper_params_grad = np.zeros([self.num_latent, self.num_hyper_params], dtype=np.float32)
        if self.num_data_points == self.num_inducing and self.cached_ell is not None:
            # The data is not sparse hence the gradient will be 0.
            return self.cached_ell, hyper_params_grad

        data_inducing_kernel, kernel_products, diag_conditional_covars = (
            self._get_interim_matrices(input_partition))
        for i in xrange(self.num_components):
            # Pre-compute values relevant to calculating the gradients and ell.
            partition_size = input_partition.shape[0]
            normal_samples, sample_means, sample_vars, samples = (
                self._get_samples_partition(i, partition_size, kernel_products, diag_conditional_covars))
            conditional_ll, _ = self.likelihood.ll_F_Y(samples, output_partition)
            conditional_ll = conditional_ll.astype(np.float32)
            # Now compute gradients and ell for this component.
            ell += self._calculate_ell(
                i, output_partition, conditional_ll, sample_means, sample_vars)
            # Increment the gradient if the data is not sparse.
            if self.num_data_points != self.num_inducing:
                hyper_params_grad += self._grad_ell_over_hyper_params(
                    i, input_partition, conditional_ll, data_inducing_kernel, kernel_products,
                    sample_vars, normal_samples)
        return ell, hyper_params_grad

    def _likelihood_params_ell(self, input_partition, output_partition):
        """
        Calculate the expected log likelihood alongside its gradients with respect to
        the likelihood parameters.

        Parameters
        ----------
        input_partition : ndarray
            The input data. Dimensions: partition_size * input_dim.
        output_partition : ndarray
            The output data. Dimensions: partition_size * output_dim.

        Returns
        -------
        ell : float
            The value of the expected log likelihood over the given data.
        hyper_params_grad : ndarray
            The gradient of the ell with respect to the likelihood parameters.
        """
        ell = 0
        likelihood_grad = np.zeros(self.num_likelihood_params, dtype=np.float32)
        data_inducing_kernel, kernel_products, diag_conditional_covars = (
            self._get_interim_matrices(input_partition))

        for i in xrange(self.num_components):
            # Pre-compute values relevant to calculating the gradients and ell.
            partition_size = input_partition.shape[0]
            _, sample_means, sample_covars, samples = (
                self._get_samples_partition(i, partition_size, kernel_products, diag_conditional_covars))
            conditional_ll, curr_grad = self.likelihood.ll_F_Y(samples, output_partition)
            conditional_ll = conditional_ll.astype(np.float32)

            # Now compute gradients and ell for this component.
            ell += self._calculate_ell(
                i, output_partition, conditional_ll, sample_means, sample_covars)
            likelihood_grad += self.gaussian_mixture.weights[i] * curr_grad.sum() / self.num_samples

        return ell, likelihood_grad

    def _inducing_ell(self, input_partition, output_partition):
        """
        Calculate the expected log likelihood alongside its gradients with respect to
        the inducing points.

        Parameters
        ----------
        input_partition : ndarray
            The input data. Dimensions: partition_size * input_dim.
        output_partition : ndarray
            The output data. Dimensions: partition_size * output_dim.

        Returns
        -------
        ell : float
            The value of the expected log likelihood over the given data.
        inducing_grad : ndarray
            The gradient of the ell with respect to the inducing points.
        """
        ell = 0
        inducing_grad = np.zeros([self.num_latent, self.num_inducing, self.input_dim],
                                 dtype=np.float32)
        data_inducing_kernel, kernel_products, diag_conditional_covars = (
            self._get_interim_matrices(input_partition))

        for i in xrange(self.num_components):
            # Pre-compute values relevant to calculating the gradients and ell.
            partition_size = input_partition.shape[0]
            normal_samples, sample_means, sample_vars, samples = (
                self._get_samples_partition(i, partition_size, kernel_products, diag_conditional_covars))
            conditional_ll, _ = self.likelihood.ll_F_Y(samples, output_partition)
            conditional_ll = conditional_ll.astype(np.float32)

            # Now compute gradients and ell for this component.
            ell += self._calculate_ell(i, output_partition, conditional_ll,
                                       sample_means, sample_vars)
            inducing_grad += self._grad_ell_over_inducing(i, input_partition, conditional_ll,
                                                          data_inducing_kernel, kernel_products,
                                                          sample_vars, normal_samples)
        return ell, inducing_grad

    def _calculate_cross(self, grad_cross_over_weights):
        """
        Calculate the current cross entropy value.

        Parameters
        ----------
        grad_cross_over_weights : ndarray
            The gradient of the cross entropy with respect to the mixture of gaussian weights.
            Dimension: num_components

        Returns
        -------
        cross : float
            The value of the cross entropy.
        """
        cross = np.float32(0)
        for i in xrange(self.num_components):
            cross += self.gaussian_mixture.weights[i] * grad_cross_over_weights[i]
        return cross

    def _grad_cross_over_means(self):
        """
        Calculate the gradient of the cross entropy with respect to the mixture of Gaussian's means.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_components * num_latent * num_inducing.
        """
        grad = np.empty([self.num_components, self.num_latent, self.num_inducing], dtype=np.float32)
        for i in xrange(self.num_components):
            for j in xrange(self.num_latent):
                grad[i, j] = -(self.gaussian_mixture.weights[i] *
                               scipy.linalg.cho_solve((self.kernel_matrix.cholesky[j], True),
                                                      (self.gaussian_mixture.means[i, j] - self.GP_mean[j])))

        return grad

    def _grad_cross_over_covars(self):
        """
        Calculate the gradient of the cross entropy with respect to the mixture of Gaussian's
        covariances.

        Returns
        -------
        grad : ndarray
            The value of the gradient.
        """
        grad = np.empty([
            self.num_components, self.num_latent, self.gaussian_mixture.get_covar_size()],
            dtype=np.float32)
        for i in xrange(self.num_components):
            for j in xrange(self.num_latent):
                grad_trace = self.gaussian_mixture.grad_trace_a_inv_dot_covars(
                    self.kernel_matrix.cholesky[j], i, j)
                grad[i, j] = (-0.5 * self.gaussian_mixture.weights[i] * grad_trace)

        return grad.flatten()

    def _grad_cross_over_weights(self):
        """
        Calculate the gradient of the cross entropy with respect to the mixture of Gaussian's
        weights.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimension: num_components.
        """
        grad = np.zeros(self.num_components, dtype=np.float32)
        for i in xrange(self.num_components):
            for j in xrange(self.num_latent):
                mean = self.gaussian_mixture.means[i, j]
                mean_dot_kern_inv_dot_mean = mdot(mean.T - self.GP_mean[j], scipy.linalg.cho_solve(
                    (self.kernel_matrix.cholesky[j], True), mean - self.GP_mean[j]))
                grad[i] += (
                    self.num_inducing * np.log(2 * np.pi) + self.kernel_matrix.log_determinant[j] +
                    mean_dot_kern_inv_dot_mean + self.gaussian_mixture.trace_with_covar(
                    self.kernel_matrix.inverse[j], i, j))

        grad *= -0.5
        return grad

    def _grad_cross_over_hyper_params(self):
        """
        Calculate the gradient of the cross entropy with respect to the kernel hyper parameters.

        Returns
        -------
        grad : ndarray
           The value of the gradient. Dimensions: num_latent * num_hyper_params.
        """
        grad = np.empty([self.num_latent, self.num_hyper_params], dtype=np.float32)
        for i in xrange(self.num_latent):
            self.kernels_latent[i].update_gradients_full(self._grad_cross_over_kernel_matrix(i),
                                                         self.inducing_locations[i])
            grad[i] = self.kernels[i].gradient.copy()

        return grad

    def _grad_cross_over_inducing(self):
        """
        Calculate the gradient of the cross entropy with respect to the inducing point locations.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent, num_inducing, input_dim.
        """
        grad = np.empty([self.num_latent, self.num_inducing, self.input_dim], dtype=np.float32)
        for i in xrange(self.num_latent):
            grad[i] = self.kernels_latent[i].gradients_X(self._grad_cross_over_kernel_matrix(i),
                                                         self.inducing_locations[i])

        return grad

    def _grad_cross_over_kernel_matrix(self, latent_index):
        """
        Calculate the gradient of the cross entropy with respect to the kernel of a latent process.

        Parameters
        ----------
        latent_index : int
            The index of the latent process with respect to which we are calculating the gradient.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_inducing * num_inducing.
        """
        grad = np.zeros([self.num_inducing, self.num_inducing], dtype=np.float32)
        for i in xrange(self.num_components):
            grad += (
                -0.5 * self.gaussian_mixture.weights[i] * (self.kernel_matrix.inverse[latent_index] -
                scipy.linalg.cho_solve((self.kernel_matrix.cholesky[latent_index], True),
                scipy.linalg.cho_solve((self.kernel_matrix.cholesky[latent_index], True),
                self.gaussian_mixture.mean_prod_sum_covar(i, latent_index, self.GP_mean[latent_index]).T).T)))

        return grad

    def _calculate_entropy(self):
        """
        Calculate the current value of the entropy.

        Returns
        -------
        ent : float
            The current value of the entropy.
        """
        raise NotImplementedError

    def _grad_entropy_over_means(self):
        """
        Calculate the gradients of the entropy with respect to the mixture of Gaussian's means.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_components * num_latent * num_inducing.
        """
        raise NotImplementedError

    def _grad_entropy_over_covars(self):
        """
        Calculate the gradient of the entropy term with respect to the mixture of Gaussian's
        covariances.

        Returns
        -------
        grad : ndarray
            The value of the gradient.
        """
        raise NotImplementedError

    def _grad_entropy_over_weights(self):
        """
        Calculate the gradient of the entropy term with respect to the mixture of Gaussian's
        weights.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimension: num_components.
        """
        raise NotImplementedError

    def _calculate_ell(self, component_index, output_partition, conditional_ll,
                       sample_means, sample_vars):
        """
        Calculate the expected log likelihood for one of the components of the mixture of Gaussians.
        If self.is_exact_ell is set we use the exact ell value provided by the likelihood function.

        Parameters
        ----------
        component_index : int
            The index of the component we are calculating the expected log likelihood for.
        output_partition : ndarray
            The output of the data over which we are calculating the expected log likelihood.
            Dimensions: num_data_points * output_dim.
        conditional_ll : ndarray
            The values log(p(y|f)) where f is approximated using samples and y is the output data.
            Dimensions: partition_size * num_samples.
        sample_means : ndarray
            The means of the normal distributions used to generate the samples of the latent
            process. Dimensions: num_latent * partition_size.
        sample_vars : ndarray
            The variances of the normal distributions used to generate the samples of the latent
            process. Dimensions: num_latent * partition_size.

        Returns
        -------
        ell : float
            The value of the expected log likelihood.
        """
        ell = np.float32(0.0)
        if self.is_exact_ell:
            for i in xrange(len(output_partition)):
                unweighted_ell = self.likelihood.ell(sample_means[:, i], sample_vars[:, i],
                                                     output_partition[i])
                ell += self.gaussian_mixture.weights[component_index] * unweighted_ell
        else:
            ell = (self.gaussian_mixture.weights[component_index] *
                   conditional_ll.sum() / self.num_samples)
        return ell.astype(np.float32)

    def _grad_ell_over_means(self, component_index, conditional_ll,
                             data_inducing_kernel, sample_vars, normal_samples):
        """
        Calculate the gradient of the expected log likelihood with respect to the mean of a specific
        mixture of Gaussian's component.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        conditional_ll : ndarray
            The values log(p(y|f)) where f is approximated using samples and y is the output data.
            Dimensions: partition_size * num_samples.
        data_inducing_kernel : ndarray
            The covariance matrix between the input partition and the inducing points.
            Dimensions: num_latent * num_inducing * partition_size.
        sample_vars : ndarray
            The variances used to generate the samples.
            Dimensions: num_latent * partition_size * num_latent.
        normal_samples : ndarray
            The normal samples used to generate the final samples.
            Dimensions: num_latent * num_samples * partition_size.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing.
        """
        grad = np.empty([self.num_latent, self.num_inducing], dtype=np.float32)
        for i in xrange(self.num_latent):
            # mean = util.weighted_average(conditional_ll, normal_samples[i] / np.sqrt(sample_vars[i]), self.num_samples)
            mean = util.average_ctrl_variates(conditional_ll, normal_samples[i] / sample_vars[i], self.num_samples)

            # TODO(karl): Figure out why we need a double mdot here.
            grad[i] = (self.gaussian_mixture.weights[component_index] *
                       scipy.linalg.cho_solve((self.kernel_matrix.cholesky[i], True),
                       mdot(mean, data_inducing_kernel[i].T)))
        return grad

    def _grad_ell_over_covars(self, component_index, conditional_ll, kernel_products, sample_vars,
                              normal_samples):
        """
        Calculate the gradient of the expected log likelihood with respect to the covariance of a
        specific mixture of Gaussian's component.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        conditional_ll : ndarray
            The values log(p(y|f)) where f is approximated using samples and y is the output data.
            Dimensions: partition_size * num_samples.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.
        sample_vars : ndarray
            The variances used to generate the samples.
            Dimensions: num_latent * partition_size * num_latent.
        normal_samples : ndarray
            The normal samples used to generate the final samples.
            Dimensions: num_latent * num_samples * partition_size.

        Returns
        -------
        grad : ndarray
            The value of the gradient.
        """
        raise NotImplementedError

    def _grad_ell_over_hyper_params(self, component_index, input_partition, conditional_ll,
                                    data_inducing_kernel, kernel_products, sample_vars,
                                    normal_samples):
        """
        Calculate the gradient of the expected log likelihood with respect to the kernel
        hyper-parameters for one mixture of Gaussian component.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        conditional_ll : ndarray
            The values log(p(y|f)) where f is approximated using samples and y is the output data.
            Dimensions: partition_size * num_samples.
        data_inducing_kernel : ndarray
            The covariance matrix between the input partition and the inducing points.
            Dimensions: num_latent * num_inducing * partition_size.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.
        sample_vars : ndarray
            The variances used to generate the samples.
            Dimensions: num_latent * partition_size * num_latent.
        normal_samples : ndarray
            The normal samples used to generate the final samples.
            Dimensions: num_latent * num_samples * partition_size.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_hyper_params.
        """
        hyper_params_grad = np.empty([self.num_latent, self.num_hyper_params], dtype=np.float32)

        for i in xrange(self.num_latent):
            grad_vars = self._grad_sample_vars_over_hyper_params(component_index, i,
                                                                 input_partition,
                                                                 data_inducing_kernel,
                                                                 kernel_products)
            grad_means = self._grad_sample_means_over_hyper_params(component_index, i,
                                                                   input_partition,
                                                                   kernel_products)

            for j in xrange(self.num_hyper_params):
                # TODO(karl): Name this something more meaningful or refactor.
                val = (np.ones(conditional_ll.shape, dtype=np.float32) / sample_vars[i] *
                       grad_vars[:, j] - 2.0 * normal_samples[i] / np.sqrt(sample_vars[i]) *
                       grad_means[:, j] - np.square(normal_samples[i]) / sample_vars[i] *
                       grad_vars[:, j])
                mean = util.average_ctrl_variates(conditional_ll, val, self.num_samples)
                hyper_params_grad[i, j] = (
                    -1.0 / 2 * self.gaussian_mixture.weights[component_index] * mean.sum())

        return hyper_params_grad

    def _grad_ell_over_inducing(self, component_index, input_partition, conditional_ll,
                                data_inducing_kernel, kernel_products, sample_vars, normal_samples):
        """
        Calculate the gradient of the expected log likelihood with respect to the inducing point
        locations for one mixture of Gaussian component.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        conditional_ll : ndarray
            The values log(p(y|f)) where f is approximated using samples and y is the output data.
            Dimensions: partition_size * num_samples.
        data_inducing_kernel : ndarray
            The covariance matrix between the input partition and the inducing points.
            Dimensions: num_latent * num_inducing * partition_size.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.
        sample_vars : ndarray
            The variances used to generate the samples.
            Dimensions: num_latent * partition_size * num_latent.
        normal_samples : ndarray
            The normal samples used to generate the final samples.
            Dimensions: num_latent * num_samples * partition_size.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing * input_dim.
        """
        grad = np.empty([self.num_latent, self.num_inducing, self.input_dim], dtype=np.float32)
        for i in xrange(self.num_latent):
            grad_means = self._grad_sample_means_over_inducing(component_index, i, input_partition,
                                                               kernel_products)
            grad_means = grad_means.reshape(grad_means.shape[0], grad_means.shape[1] *
                                            grad_means.shape[2])
            grad_vars = self._grad_sample_vars_over_inducing(component_index, i, input_partition,
                                                             data_inducing_kernel, kernel_products)
            grad_vars = grad_vars.reshape(grad_vars.shape[0], grad_vars.shape[1] *
                                          grad_vars.shape[2])

            raw_gradient = self._theano_grad_ell_over_inducing(
                self.gaussian_mixture.weights[component_index], grad_means, grad_vars,
                conditional_ll, normal_samples[i], sample_vars[i])
            grad[i] = raw_gradient.reshape([self.num_inducing, self.input_dim])

        return grad

    def _compile_grad_ell_over_inducing():
        weight = tensor.scalar('weight')
        grad_means = tensor.matrix('grad_means')
        grad_vars = tensor.matrix('grad_vars')
        conditional_ll = tensor.matrix('conditional_ll')
        normal_samples = tensor.matrix('normal_samples')
        sample_vars = tensor.vector('sample_vars')

        raw_gradient = (
            tensor.dot(conditional_ll / sample_vars, grad_vars) - 2.0 *
            tensor.dot(conditional_ll * normal_samples / tensor.sqrt(sample_vars), grad_means) -
            tensor.dot(conditional_ll * tensor.sqr(normal_samples) / sample_vars, grad_vars))
        gradient = -0.5 * weight * raw_gradient.mean(axis=0)

        return theano.function([weight, grad_means, grad_vars, conditional_ll, normal_samples,
                               sample_vars], gradient)
    _theano_grad_ell_over_inducing = _compile_grad_ell_over_inducing()

    def _grad_sample_means_over_hyper_params(self, component_index, latent_index, input_partition,
                                             kernel_products):
        """
        Calculate the gradient of the sample means with respect to the kernel hyper-parameters
        for one mixture of Gaussian component and one latent process.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        latent_index : int
            The index of the latent process with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing * input_dim.
        """
        repeated_means = np.repeat(
            self.gaussian_mixture.means[component_index, latent_index][:, np.newaxis],
            input_partition.shape[0], axis=1) - self.GP_mean[latent_index]
        return self._grad_kernel_product_over_hyper_params(latent_index, input_partition,
                                                           kernel_products, repeated_means)

    def _grad_sample_vars_over_hyper_params(self, component_index, latent_index, input_partition,
                                            data_inducing_kernel, kernel_products):
        """
        Calculate the gradient of the sample variances with respect to the kernel hyper-parameters
        for one mixture of Gaussian component and one latent process.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        latent_index : int
            The index of the latent process with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        data_inducing_kernel : ndarray
            The covariance matrix between the input partition and the inducing points.
            Dimensions: num_latent * num_inducing * partition_size.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing * input_dim.
        """
        # TODO(karl): Fix the naming.
        input_kernel_grad = self.kernels[latent_index].get_gradients_Kdiag(input_partition)
        kernel_products_grad = self.kernels[latent_index].get_gradients_AK(
            kernel_products[latent_index], input_partition, self.inducing_locations[latent_index])
        covar_product = self.gaussian_mixture.covar_dot_a(
            kernel_products[latent_index].T, component_index, latent_index)
        grad_product = self._grad_kernel_product_over_hyper_params(
            latent_index, input_partition, kernel_products,
            covar_product - data_inducing_kernel[latent_index] / 2)

        return input_kernel_grad - kernel_products_grad + 2.0 * grad_product

    def _grad_sample_means_over_inducing(self, component_index, latent_index, input_partition,
                                         kernel_products):
        """
        Calculate the gradient of the sample means with respect to the inducing point locations
        for one mixture of Gaussian component and one latent process.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        latent_index : int
            The index of the latent process with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing * input_dim.
        """
        repeated_means = np.repeat(
            self.gaussian_mixture.means[component_index, latent_index][:, np.newaxis],
            input_partition.shape[0], axis=1)
        return self._grad_kernel_product_over_inducing(latent_index, input_partition,
                                                       kernel_products, repeated_means)

    def _grad_sample_vars_over_inducing(self, component_index, latent_index, input_partition,
                                        data_inducing_kernel, kernel_products):
        """
        Calculate the gradient of the sample variances with respect to the kernel hyper-parameters
        for one mixture of Gaussian component and one latent process.

        Parameters
        ----------
        component_index : int
            The index of the component with respect to which we wish to calculate the gradient.
        latent_index : int
            The index of the latent process with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        data_inducing_kernel : ndarray
            The covariance matrix between the input partition and the inducing points.
            Dimensions: num_latent * num_inducing * partition_size.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing * input_dim.
        """
        # TODO(karl): Fix the naming, investigate memory efficiency.
        temp1 = -self.kernels[latent_index].get_gradients_X_AK(
            kernel_products[latent_index].T, self.inducing_locations[latent_index], input_partition)
        temp2 = (
            self.gaussian_mixture.covar_dot_a(kernel_products[latent_index].T, component_index,
            latent_index) - data_inducing_kernel[latent_index] / 2.0)
        temp1 += 2.0 * self._grad_kernel_product_over_inducing(latent_index, input_partition,
            kernel_products, temp2)
        return temp1

    def _grad_kernel_product_over_hyper_params(self, latent_index, input_partition, kernel_products,
                                               m):
        """
        Calculate the gradient of the kernel products with respect to the hyper-parameters for one
        latent process.

        Parameters
        ----------
        latent_index : int
            The index of the latent process with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing * input_dim.
        """
        # TODO(karl): Consider removing m.
        w = scipy.linalg.cho_solve((self.kernel_matrix.cholesky[latent_index], True), m)
        return (self.kernels[latent_index].get_gradients_AK(w.T, input_partition,
                self.inducing_locations[latent_index]) -
                self.kernels[latent_index].get_gradients_SKD(kernel_products[latent_index], w,
                self.inducing_locations[latent_index]))

    def _grad_kernel_product_over_inducing(self, latent_index, input_partition, kernel_products, m):
        """
        Calculate the gradient of the kernel products with respect to the inducing point locations
        for one latent process.

        Parameters
        ----------
        latent_index : int
            The index of the latent process with respect to which we wish to calculate the gradient.
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.

        Returns
        -------
        grad : ndarray
            The value of the gradient. Dimensions: num_latent * num_inducing * input_dim.
        """
        # TODO(karl): Consider removing m. Optimize memory. Rename vars.
        w = scipy.linalg.cho_solve((self.kernel_matrix.cholesky[latent_index], True), m)
        temp1 = self.kernels[latent_index].get_gradients_X_AK(
            w, self.inducing_locations[latent_index], input_partition)
        temp2 = self.kernels[latent_index].get_gradients_X_SKD(
            kernel_products[latent_index], w, self.inducing_locations[latent_index])
        return temp1 - temp2

    def _get_interim_matrices(self, input_partition):
        """
        Get matrices that are used as intermediate values in various calculations.

        Parameters
        ----------
        input_partition : ndarray
            A partition of the input data. Dimensions: partition_size * input_dim.

        Returns
        -------
        data_inducing_kernel : ndarray
            The covariance matrix between the input partition and the inducing points.
            Dimensions: num_latent * num_inducing * partition_size.
        kernel_products : ndarray
            The product of kernel_matrix and data_inducing_kernel.
            Dimensions: num_latent * partition_size * num_inducing.
        diag_conditional_covars : ndarray
            The diagonal of the covariance of p(f|u) for each latent process f and inducing point u.
            Dimensions: num_latent * partition_size.
        """
        partition_size = input_partition.shape[0]
        data_inducing_kernel = np.empty([self.num_latent, self.num_inducing, partition_size],
                                        dtype=np.float32)
        kernel_products = np.empty([self.num_latent, partition_size, self.num_inducing],
                                   dtype=np.float32)
        diag_conditional_covars = np.empty([self.num_latent, partition_size], dtype=np.float32)

        for j in xrange(self.num_latent):
            data_inducing_kernel[j] = self.kernels[j].kernel(self.inducing_locations[j],
                                                             input_partition)
            kernel_products[j] = scipy.linalg.cho_solve((self.kernel_matrix.cholesky[j], True),
                                                        data_inducing_kernel[j]).T
            diag_conditional_covars[j] = (self.kernels[j].diag_kernel(input_partition) -
                np.sum(kernel_products[j] * data_inducing_kernel[j].T, 1))
        return data_inducing_kernel, kernel_products, diag_conditional_covars

    def _get_samples_partition(self, component_index, partition_size, kernel_products,
                               diag_conditional_covars):
        """
        Get samples used to approximate latent process values and information about them.
        For each data point in the partition, num_samples get generated.

        Parameters
        ----------
        component_index : int
            The mixture of Gaussians component we wish to get sample data for.
        partition_size : int
            The size of the data partition for which we are generating samples.
        kernel_products : ndarray
            The product between two kernel matrices. See get_interim_matrices for details.
            Dimensions: num_latent * partition_size * num_inducing.
        diag_conditional_covars : ndarray
            The diagonal of the covariance of p(f|u) for each latent process f and inducing point u.
            Dimensions: num_latent * partition_size.

        Returns
        -------
        normal_samples : ndarray
            The normal samples used to generate the final samples.
            Dimensions: num_latent * num_samples * partition_size.
        sample_means : ndarray
            The means of the samples to generate. Dimensions: num_latent * partition_size.
        sample_vars : ndarray
            The variances of the samples to generate.
            Dimensions: num_latent * partition_size * num_latent.
        samples : ndarray
            The generated samples. Dimensions: num_samples * partition_size * num_latent.
        """
        normal_samples = np.empty([self.num_latent, self.num_samples, partition_size],
                                  dtype=np.float32)
        sample_means = np.empty([self.num_latent, partition_size], dtype=np.float32)
        sample_vars = np.empty([self.num_latent, partition_size], dtype=np.float32)
        samples = np.empty([self.num_samples, partition_size, self.num_latent], dtype=np.float32)

        for i in xrange(self.num_latent):
            kern_dot_covar_dot_kern = self.gaussian_mixture.a_dot_covar_dot_a(kernel_products[i],
                                                                              component_index, i)

            # non-zero mean GP prior chang: b = fbar + (m - mubar)
            m_u = self.gaussian_mixture.means[component_index, i] - self.GP_mean[i]
            normal_samples[i], sample_means[i], sample_vars[i], samples[:, :, i] = (
                self._theano_get_samples_partition(kernel_products[i],
                                         diag_conditional_covars[i],
                                         kern_dot_covar_dot_kern,
                                         m_u,
                                         self.num_samples))
            sample_means[i] =  sample_means[i] + self.GP_mean[i] # non-zero mean GP prior chang: b = fbar + (m - mubar)
            samples[:, :, i] = samples[:, :, i] + self.GP_mean[i] # non-zero mean GP prior chang: b = fbar + (m - mubar)
        return normal_samples, sample_means, sample_vars, samples

    def _compile_get_samples_partition():
        random_stream = rng_mrg.MRG_RandomStreams(seed=1)

        kernel_products = tensor.matrix('kernel_products')
        diag_conditional_covars = tensor.vector('diag_conditional_covars')
        kern_dot_covars_dot_kern = tensor.vector('kern_dot_covars_dot_kern')
        gaussian_mixture_means = tensor.vector('gaussian_mixture_means')
        num_samples = tensor.iscalar('num_samples')
        partition_size = kernel_products.shape[0]

        normal_samples = random_stream.normal([num_samples, partition_size])
        sample_means = tensor.dot(kernel_products, gaussian_mixture_means.T)
        sample_vars = diag_conditional_covars + kern_dot_covars_dot_kern
        samples = normal_samples * tensor.sqrt(sample_vars) + sample_means

        return theano.function([kernel_products, diag_conditional_covars, kern_dot_covars_dot_kern,
                                gaussian_mixture_means, num_samples], [normal_samples, sample_means,
                                sample_vars, samples])
    _theano_get_samples_partition = _compile_get_samples_partition()

    def _predict_partition(self, input_partition, output_partition):
        """
        Predict the output value of a given input partition, and if output_partition is given also
        calculate the negative log predictive density. Predictions are made over each component.

        Parameters
        ----------
        input_partition: ndarray
            Test point inputs. Dimensions: partition_size * input_dim.
        output_partition: ndarray (or None)
            Test point outputs. Dimensions: partition_size * output_dim.

        Returns
        -------
        predicted_means: ndarray
            The expected value of p(Y|X) where Y is the output partition and X is the input
            partition. Dimensions: partition_size * num_components * output_dim.
        predicted_vars: ndarray
            The variance of p(Y|X). Dimensions: partition_size * num_components * output_dim.
        nlpd: ndarray
            The negative log of p(Y|X). Dimensions: partition_size * nlpd_dim * num_components.
        """
        partition_size = input_partition.shape[0]
        predicted_means = np.empty([
            partition_size, self.num_components, self.likelihood.output_dim()], dtype=np.float32)
        predicted_vars = np.empty([
            partition_size, self.num_components, self.likelihood.output_dim()], dtype=np.float32)
        nlpd = np.empty([partition_size, self.likelihood.nlpd_dim(), self.num_components],
                        dtype=np.float32)
        data_inducing_kernel, kernel_products, diag_conditional_covars = (
            self._get_interim_matrices(input_partition))

        for i in xrange(self.num_components):
            _, sample_means, sample_vars, _ = self._get_samples_partition(
                i, partition_size, kernel_products, diag_conditional_covars)
            predicted_means[:, i], predicted_vars[:, i], nlpd[:, :, i] = (
                self.likelihood.predict(sample_means.T, sample_vars.T, output_partition, self))

        if output_partition is not None:
            nlpd = -scipy.misc.logsumexp(nlpd, 2, self.gaussian_mixture.weights)
        else:
            nlpd = None

        return predicted_means, predicted_vars, nlpd

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['kernel_matrix']
        return result

    def __setstate__(self, dict):
        self.__dict__ = dict
        self.kernel_matrix = util.PosDefMatrix(self.num_latent, self.num_inducing)
        self.kernel_matrix.update(self.kernels, self.inducing_locations)
