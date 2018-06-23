from GPy.util.linalg import mdot
import numpy as np

import gaussian_mixture
import util


class DiagonalGaussianMixture(gaussian_mixture.GaussianMixture):
    """
    Implementation of a posterior distribution where the covariance matrix is a mixture of diagonal Gaussians.
    The class has to important internal field as follows:

     Attributes
     ----------
     log_s : ndarray
      Logarithm of the diagonal of covariance matrix

     invC_klj_Sk : ndarray
      (s[k,j] + s[l,j])^-1 * s[k,j]
    """
    def __init__(self, num_components, num_latent, initial_mean):
        gaussian_mixture.GaussianMixture.__init__(self, num_components, num_latent, initial_mean)
        self.invC_klj_Sk = np.empty((self.num_components, self.num_components, self.num_latent, self.num_dim), dtype=util.PRECISION)
        self.covars = np.random.uniform(low=0.5, high=0.5, size=(self.num_components, self.num_latent, self.num_dim)).astype(util.PRECISION)
        self.log_s = np.log(self.covars)
        self._update()

    def get_params(self):
        return np.hstack([self.means.flatten(), self.log_s.flatten(),
                          np.log(self.unnormalized_weights)])

    def transform_covars_grad(self, g):
        r"""
        Assume:
        g = df \\ dS

        then this function returns:
        :returns df \\ d log(s)

        therefore transforming the gradient to the raw space (log(s) space).
        """
        return g.flatten() * self.covars.flatten()

    def get_covar_size(self):
        return self.num_dim

    def get_covar_shape(self):
        return [self.num_dim]

    def set_covars(self, sa):
        self.covars = np.exp(sa).reshape([self.num_components, self.num_latent, self.num_dim])
        self.log_s = sa.reshape([self.num_components, self.num_latent, self.num_dim])

    def trace_with_covar(self, A, k, j):
        return np.dot(np.diagonal(A), self.covars[k, j])

    def C_m(self, j, k, l):
        """
        Returns (m[k,j] - m[l,j]) / (s[l,j] + s[k,j])
        """
        return (self.means[k, j, :] - self.means[l, j, :]) / (self.covars[l, j, :] + self.covars[k, j, :])

    def C_m_C(self, j, k, l):
        """
        Returns (1 / (s[k,j] + s[l,j]) - (m[k,j] - m[l,j]) ** 2 / (s[k,j] + s[l,j])) * s[k,j]

        None that the last multiplication by s[k,j] is because this function is used to calculate
        gradients, and this multiplication brings the gradients to the raw space (log(s) space)
        """
        return (self.invC_klj_Sk[k, l, j] -
                np.square(self.invC_klj_Sk[k, l, j] * (self.means[k, j] - self.means[l, j])) / self.covars[k, j])

    def a_dot_covar_dot_a(self, a, k, j):
        return np.diagonal(mdot(a, np.diag(self.covars[k, j, :]), a.T))

    def mean_prod_sum_covar(self, k, j):
        return mdot(self.means[k, j, np.newaxis].T, self.means[k, j, np.newaxis]) + np.diag(self.covars[k, j])

    def grad_trace_a_inv_dot_covars(self, chol_a, k, j):
        a_inv = util.inv_chol(chol_a)
        return np.diagonal(a_inv) * self.covars[k, j, :].flatten()

    def covar_dot_a(self, a, k, j):
        return mdot(np.diag(self.covars[k, j]), a)

    def _update(self):
        for k in range(self.num_components):
            for l in range(self.num_components):
                for j in range(self.num_latent):
                    self.invC_klj_Sk[k,l,j] = self._s_k_skl(k,l,j)

    def _s_k_skl(self, k, l, j):
        """
        calculates s[k,j] / (s[k,j] + s[k,l]) in a hopefully numerical stable manner.
        """

        a = np.maximum(self.log_s[k, j, :], self.log_s[l, j, :])
        return np.exp((-a + self.log_s[k, j, :])) / (np.exp((-a + self.log_s[l, j, :])) + np.exp((-a + self.log_s[k, j, :])))

    def get_means_and_covars(self):
        return self.means.copy(), self.log_s.copy()
