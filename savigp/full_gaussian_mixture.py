"""This module implements the FullGaussianMixture class."""
from GPy.util.linalg import mdot
import numpy as np
import scipy

import gaussian_mixture
import theano
from theano import tensor
import util


class FullGaussianMixture(gaussian_mixture.GaussianMixture):
    """
    Represents a full mixture of Gaussians with a single component.

    This class keeps the covariance parameters up to date and provides various utility functions
    that allow the FullGaussianProcess class to operate on its posterior parameters.

    Parameters
    ----------
    num_latent : int
        The number of latent processes that the mixture outputs to.
    initial_mean : ndarray
        The initial mean of the gaussian mixture. Dimensions: 1 * num_latent * num_dim.
    """
    def __init__(self, num_latent, initial_mean, init_var):
        super(FullGaussianMixture, self).__init__(1, num_latent, initial_mean)
        self.covars_cholesky = np.tile(np.eye(self.num_dim, dtype=np.float32),
                                       [self.num_latent, 1, 1])
        if init_var is not None:
            self.covars_cholesky = np.sqrt(init_var[:, np.newaxis, np.newaxis]) * self.covars_cholesky
            self.covars = init_var[:, np.newaxis, np.newaxis] * self.covars_cholesky
        else:
            self.covars = self.covars_cholesky.copy()

    def get_params(self):
        return np.hstack([self.means.flatten(), self._get_raw_covars(),
                          np.log(self.unnormalized_weights)])

    def get_means_and_covars(self):
        return self.means.copy(), self._get_raw_covars()

    def set_covars(self, raw_covars):
        raw_covars = raw_covars.reshape([self.num_latent, self.get_covar_size()])
        for j in xrange(self.num_latent):
            cholesky = np.zeros([self.num_dim, self.num_dim], dtype=np.float32)
            cholesky[np.tril_indices_from(cholesky)] = raw_covars[j]
            cholesky[np.diag_indices_from(cholesky)] = np.exp(
                cholesky[np.diag_indices_from(cholesky)])
            self.covars_cholesky[j] = cholesky
            self.covars[j] = mdot(self.covars_cholesky[j], self.covars_cholesky[j].T)

    def log_normal(self):
        log_normal = -0.5 * (self.num_latent * self.num_dim * np.log(2 * np.pi) + np.log(2))
        for i in xrange(self.num_latent):
            log_normal -= 0.5 * util.pddet(self.covars_cholesky[i])
        return log_normal.astype(np.float32)

    def a_dot_covar_dot_a(self, a, component_index, latent_index):
        return self._theano_a_dot_covar_dot_a(a, self.covars[latent_index])

    def _compile_a_dot_covar_dot_a():
        a = tensor.matrix('a')
        covar = tensor.matrix('covar')

        result = tensor.sum(a * tensor.dot(a, covar), 1)
        return theano.function([a, covar], result)
    _theano_a_dot_covar_dot_a = _compile_a_dot_covar_dot_a()

    def mean_prod_sum_covar(self, component_index, latent_index, offset=0):
        assert component_index == 0
        mu = self.means[0, latent_index, :, np.newaxis] - offset
        return mdot(mu, mu.T) + self.covars[latent_index]

    def covar_dot_a(self, a, component_index, latent_index):
        assert component_index == 0
        return mdot(self.covars[latent_index], a)

    def transform_eye_grad(self):
        grad = np.empty([self.num_latent, self.get_covar_size()], dtype=np.float32)
        meye = np.eye(self.num_dim, dtype=np.float32)[np.tril_indices_from(self.covars_cholesky[0])]
        for j in range(self.num_latent):
            grad[j] = meye
        return grad.flatten()

    def get_covar_size(self):
        return self.num_dim * (self.num_dim + 1) / 2

    def get_covar_shape(self):
        return [self.num_dim, self.num_dim]

    def trace_with_covar(self, A, component_index, latent_index):
        assert component_index == 0
        return util.tr_AB(A, self.covars[latent_index])

    def grad_trace_a_inv_dot_covars(self, chol_a, component_index, latent_index):
        assert component_index == 0
        # TODO(karl): There is a bug here related to double counting.
        tmp = 2.0 * scipy.linalg.cho_solve((chol_a, True),
                                           self.covars_cholesky[latent_index])
        tmp[np.diag_indices_from(tmp)] *= (
            self.covars_cholesky[latent_index][np.diag_indices_from(tmp)])
        return tmp[np.tril_indices_from(self.covars_cholesky[latent_index])]

    def transform_covars_grad(self, internal_grad):
        grad = np.empty((self.num_latent, self.get_covar_size()), dtype=np.float32)
        for j in range(self.num_latent):
            tmp = self._theano_transform_covars_grad(internal_grad[0, j],
                                                     self.covars_cholesky[j])
            tmp[np.diag_indices_from(tmp)] *= self.covars_cholesky[j][np.diag_indices_from(tmp)]
            grad[j] = tmp[np.tril_indices_from(self.covars_cholesky[j])]
        return grad.flatten()

    def _compile_transform_covars_grad():
        internal_grad = tensor.matrix('internal_grad')
        covars_cholesky = tensor.matrix('covars_cholesky')
        result = tensor.dot(internal_grad + internal_grad.T, covars_cholesky)
        return theano.function([internal_grad, covars_cholesky], result)
    _theano_transform_covars_grad = _compile_transform_covars_grad()

    def _get_raw_covars(self):
        flattened_covars = np.empty([self.num_latent, self.get_covar_size()], dtype=np.float32)
        for i in xrange(self.num_latent):
            raw_covars = self.covars_cholesky[i].copy()
            raw_covars[np.diag_indices_from(raw_covars)] = np.log(
                raw_covars[np.diag_indices_from(raw_covars)])
            flattened_covars[i] = raw_covars[np.tril_indices_from(raw_covars)]
        return flattened_covars.flatten()


