"""This module contains the implementation of the full gaussian process class."""
from GPy.util.linalg import mdot
import numpy as np
import torch

import full_gaussian_mixture
import gaussian_process
import util


class FullGaussianProcess(gaussian_process.GaussianProcess):
    """
    The concrete class containing the implementation of scalable variational inference of gaussian
    processes where the posterior distribution consists of a single component gaussian.
    """
    def __init__(self,
                 train_inputs,
                 train_outputs,
                 num_inducing,
                 num_samples,
                 kernels,
                 likelihood,
                 latent_noise=0,
                 exact_ell=False,
                 inducing_on_inputs=False,
                 num_threads=1,
                 partition_size=3000):
        super(FullGaussianProcess, self).__init__(train_inputs=train_inputs,
                                                  train_outputs=train_outputs,
                                                  num_inducing=num_inducing,
                                                  num_components=1,
                                                  num_samples=num_samples,
                                                  kernels=kernels,
                                                  likelihood=likelihood,
                                                  latent_noise=latent_noise,
                                                  exact_ell=exact_ell,
                                                  inducing_on_inputs=inducing_on_inputs,
                                                  num_threads=num_threads,
                                                  partition_size=partition_size)

    def _get_gaussian_mixture(self, initial_mean):
        return full_gaussian_mixture.FullGaussianMixture(self.num_latent, initial_mean)

    def _grad_cross_over_covars(self):
            grad = np.empty([self.num_components, self.num_latent,
                             self.gaussian_mixture.get_covar_size()], dtype=util.PRECISION)
            for j in xrange(self.num_latent):
                grad_trace = self.gaussian_mixture.grad_trace_a_inv_dot_covars(
                    self.kernel_matrix.cholesky[j], 0, j)
                grad[0, j] = (-0.5 * grad_trace)

            return grad.flatten()

    def _grad_ell_over_covars(self, component_index, conditional_ll, kernel_products,
                              sample_vars, normal_samples):
        assert (component_index == 0)
        grad = np.empty([self.num_latent] + self.gaussian_mixture.get_covar_shape(),
                        dtype=util.PRECISION)
        for i in xrange(self.num_latent):
            average = util.weighted_average(
                conditional_ll, (normal_samples[i] ** 2 - 1) / sample_vars[i], self.num_samples)
            grad[i] = self._torch_grad_ell_over_covars(kernel_products[i], average)

        return grad

    @util.torchify
    def _torch_grad_ell_over_covars(self, kernel_products, average):
        return 0.5 * (kernel_products.t() * average).mm(kernel_products)

    def _calculate_entropy(self):
        return -self.gaussian_mixture.log_normal()

    def _grad_entropy_over_means(self):
        return np.zeros([self.num_components, self.num_latent, self.num_inducing], dtype=util.PRECISION)

    def _grad_entropy_over_covars(self):
        return self.gaussian_mixture.transform_eye_grad()

    def _grad_entropy_over_weights(self):
        return -self.gaussian_mixture.log_normal() - 1

