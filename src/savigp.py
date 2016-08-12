"""
A thin wrapper around the GaussianProcess class and functions found in optimizer.py that
provides a scikit-like API.
"""

import model_logging
import optimizer
from diagonal_gaussian_process import DiagonalGaussianProcess
from full_gaussian_process import FullGaussianProcess

class Savigp(object):
    def __init__(self,
                 likelihood,
                 kernels,
                 posterior='full',
                 num_inducing=100,
                 random_inducing=False,
                 num_samples=2000,
                 latent_noise=0.001,
                 partition_size=1000,
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

    def fit(train_inputs,
            train_outputs,
            optimization_config = None,
            optimize_stochastic=True,
            param_threshold=1e-3,
            objective_threshold=1e-5,
            max_iterations=200):
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
        elif self.posterior == 'mix1' or self.posterior == 'mix2':
            num_components = 1 if method == 'mix1' else 2
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
                                                 partition_size=partition_size)
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

    def predict(test_inputs, test_outputs=None):
        return self.model.predict(test_inputs, test_outputs)
