"""
A thin wrapper around the GaussianProcess class and functions found in optimizer.py that
provides a scikit-like API.
"""

import model_logging
import optimizer
from diagonal_gaussian_process import DiagonalGaussianProcess
from full_gaussian_process import FullGaussianProcess

class Savigp(object):
    """
    A scikit-like class which allows for predictions using scalable automated variational
    inference on Gaussian process models.

    Parameters
    ----------
    likelihood : subclass of Likelihood
        An object representing the likelihood function. See likelihood.py for the methods
        such an object needs to implement.
    kernels : list of kernels
        A list containing a kernel for each latent process. See kernel.py for the methods each
        element in the list needs to implement.
    posterior : str
        A string denoting the shape of the posterior distribution. Possible values include:
        'full' (a single Gaussian with a full covariance matrix), 'mix1' (a single Gaussian
        with a diagonal covariance matrix), and 'mix2' (a mixture of two Gaussians with
        diagonal covariance matrices).
    num_inducing : int
        The number of inducing points to use in training.
    random_inducing: boolean
        Whether to put inducing points randomly on training data. If False, inducing points will be
        determined using clustering.
    num_samples : int
        The number of samples used to approximate gradients and objective functions.
    latent_noise : float
        The amount of latent noise that will be added to each kernel.
    partition_size : int
        How large each partition of the training data will be when calculating expected log
        likelihood. In the case of batch optimization this is simply used to save on memory;
        smaller partitions lead to less memory usage but slower runtime. In the case of
        stochastic optimization, this will also denote the size of the mini-batches used in
        training.
    denbug_output : bool
        Whether to print the value of the objective function as it gets optimized.
    """
    def __init__(self,
                 likelihood,
                 kernels,
                 posterior='full',
                 num_components=1,
                 num_inducing=100,
                 random_inducing=False,
                 num_samples=2000,
                 latent_noise=0.001,
                 partition_size=100,
                 debug_output=False):
        # Init the logger and disable output if necessary.
        model_logging.init_logger('savigp', False)
        if not debug_output:
            model_logging.disable_logger()

        # Save the model configuration options.
        self.likelihood = likelihood
        self.kernels = kernels
        self.posterior = posterior
        self.num_inducing = num_inducing
        self.random_inducing = random_inducing
        self.num_samples = num_samples
        self.latent_noise = latent_noise
        self.partition_size = partition_size

        # We delay initializing the model until the user fits the data since
        # it unfortunately needs access to training data.
        self.model = None

    def fit(self,
            train_inputs,
            train_outputs,
            optimization_config = None,
            optimize_stochastic=True,
            param_threshold=1e-3,
            objective_threshold=5,
            max_iterations=200):
        """
        train_inputs : ndarray
            A matrix containing the input training data. Dimensions: num_data_points * input_dim.
        train_outputs : ndarray
            A matrix containing the output training data. Dimensions: num_data_points * num_latent.
        optimization_config : dict
            The maximum number of function evaluations per subset of parameters for each local
            optimization. Valid keys are: 'mog' (variational parameters), 'hyp' (kernel
            hyperparameters), 'll' (likelihood parameters), 'inducing' (inducing point locations).
            If a set of parameters is not included it will not get optimized.
        optimize_stochastic : bool
            Whether to optimize the model stochastically. If set to false batch optimization will
            be used.
        param_threshold : float
            The tolerance threshold for convergence of the variational parameters. The model
            is considered to have converged if (means_difference + covariances_difference) / 2 <
            param_threshold.
        objective_threshold : float
            The tolerance threshold for convergence of the objective function.
        max_iterations : int
            The maximum number of global optimisations. If max_iterations is None the model will
            keep getting optimized until convergence.
        """
        # Initialize the model. We should move this to init once we fix the API design issue.
        if self.posterior == 'full':
            self.model = FullGaussianProcess(train_inputs,
                                             train_outputs,
                                             self.num_inducing,
                                             self.num_samples,
                                             self.kernels,
                                             self.likelihood,
                                             self.latent_noise,
                                             False,
                                             self.random_inducing,
                                             num_threads=1,
                                             partition_size=self.partition_size)
        elif self.posterior == 'diag':
            self.model = DiagonalGaussianProcess(train_inputs,
                                                 train_outputs,
                                                 self.num_inducing,
                                                 num_components,
                                                 self.num_samples,
                                                 self.kernels,
                                                 self.likelihood,
                                                 self.latent_noise,
                                                 False,
                                                 self.random_inducing,
                                                 num_threads=1,
                                                 partition_size=self.partition_size)
        else:
            assert False

        # Setup default optimization values.
        if optimization_config is None:
            if optimize_stochastic:
                optimization_config = {'mog': 15, 'hyp': 5, 'll': 5, 'inducing': 2}
            else:
                optimization_config = {'mog': 50, 'hyp': 15, 'll': 15, 'inducing': 5}
        optimize_func = (optimizer.stochastic_optimize_model if optimize_stochastic else
                         optimizer.batch_optimize_model)

        # Optimize the model.
        _, total_evals = optimize_func(self.model,
                                       optimization_config,
                                       max_iterations,
                                       param_threshold,
                                       objective_threshold)
        return total_evals

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
            provided. Dimensions: num_test * output_dim.
        """

        return self.model.predict(test_inputs, test_outputs)
