"""This module contains the implementation of the full gaussian process class."""
from GPy.util.linalg import mdot
import numpy as np

import full_gaussian_mixture
import gaussian_process
import theano
from theano import tensor
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
                 partition_size=3000,
                 GP_mean=None,
                 init_var=None
                 ):
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
                                                  partition_size=partition_size,
                                                  GP_mean=GP_mean,
                                                  init_var=init_var
                                                  )

    def _get_gaussian_mixture(self, initial_mean, init_var=None):
        return full_gaussian_mixture.FullGaussianMixture(self.num_latent, initial_mean, init_var)

    def _grad_cross_over_covars(self):
            grad = np.empty([self.num_components, self.num_latent,
                             self.gaussian_mixture.get_covar_size()], dtype=np.float32)
            for j in xrange(self.num_latent):
                grad_trace = self.gaussian_mixture.grad_trace_a_inv_dot_covars(
                    self.kernel_matrix.cholesky[j], 0, j)
                grad[0, j] = (-0.5 * grad_trace)

            return grad.flatten()

    def _grad_ell_over_covars(self, component_index, conditional_ll, kernel_products,
                              sample_vars, normal_samples):
        assert (component_index == 0)
        grad = np.empty([self.num_latent] + self.gaussian_mixture.get_covar_shape(),
                        dtype=np.float32)
        for i in xrange(self.num_latent):
            average = util.average_ctrl_variates(conditional_ll, (normal_samples[i] ** 2 - 1) / sample_vars[i], self.num_samples)
            grad[i] = self._theano_grad_ell_over_covars(kernel_products[i], average)

        return grad

    def _compile_grad_ell_over_covars():
        kernel_products = tensor.matrix('kernel_products')
        average = tensor.vector('average')
        result = 0.5 * tensor.dot(kernel_products.T * average, kernel_products)
        return theano.function([kernel_products, average], result)
    _theano_grad_ell_over_covars = _compile_grad_ell_over_covars()

    def _calculate_entropy(self):
        return -self.gaussian_mixture.log_normal()

    def _grad_entropy_over_means(self):
        return np.zeros([self.num_components, self.num_latent, self.num_inducing], dtype=np.float32)

    def _grad_entropy_over_covars(self):
        return self.gaussian_mixture.transform_eye_grad()

    def _grad_entropy_over_weights(self):
        return -self.gaussian_mixture.log_normal() - 1

