from GPy.util.linalg import mdot
from scipy.misc import logsumexp
from util import average_ctrl_variates, log_diag_gaussian
import numpy as np


import diagonal_gaussian_mixture
import gaussian_process


class DiagonalGaussianProcess(gaussian_process.GaussianProcess):
    """
    Implementation of the SAVIGP model in the case that posterior is the mixture of diagonal Gaussians.
    """

    def __init__(self, train_inputs, train_outputs, num_inducing, num_components, num_samples, kernels,
                 likelihood, latent_noise=0, exact_ell=False, inducing_on_inputs=False,
                 num_threads=1, partition_size=3000, GP_mean=None, init_var=None):
        super(DiagonalGaussianProcess, self).__init__(
                                                      train_inputs, train_outputs, num_inducing,
                                                      num_components, num_samples, kernels, likelihood,
                                                      latent_noise, exact_ell, inducing_on_inputs,
                                                      num_threads, partition_size, GP_mean=GP_mean, init_var=init_var)

    def _get_gaussian_mixture(self, initial_mean):
        return diagonal_gaussian_mixture.DiagonalGaussianMixture(
            self.num_components, self.num_latent, initial_mean, init_var=None)

    def _grad_ell_over_covars(self, component_index, conditional_ll, kernel_products, sample_vars, normal_samples):
        grad = np.empty([self.num_latent] + self.gaussian_mixture.get_covar_shape())
        for i in xrange(self.num_latent):
            s = average_ctrl_variates(conditional_ll, (np.square(normal_samples[i]) - 1) / sample_vars[i], self.num_samples)
            grad[i] = (mdot(s, np.square(kernel_products[i])) * self.gaussian_mixture.weights[component_index] / 2.)
        return grad

    def update_N_z(self):
        self.log_z = np.zeros(self.num_components)
        self.log_N_kl = np.zeros([self.num_components, self.num_components])
        for k in range(self.num_components):
            for l in range(self.num_components):
                for j in range(self.num_latent):
                    self.log_N_kl[k, l] += log_diag_gaussian(self.gaussian_mixture.means[k, j], self.gaussian_mixture.means[l, j],
                                                             logsumexp(
                                                                 [self.gaussian_mixture.log_s[k, j, :], self.gaussian_mixture.log_s[l, j, :]],
                                                                 axis=0))
            self.log_z[k] = logsumexp(self.log_N_kl[k, :] + np.log(self.gaussian_mixture.weights))

    def _update_log_likelihood(self):
        self.update_N_z()
        gaussian_process.GaussianProcess._update_log_likelihood(self)

    def _grad_entropy_over_means(self):
        dent_dm = np.empty([self.num_components, self.num_latent, self.num_inducing])
        for k in range(self.num_components):
            for j in range(self.num_latent):
                dent_dm[k, j, :] = self._d_ent_d_m_kj(k, j)
        return dent_dm

    def _d_ent_d_m_kj(self, k, j):
        r"""
        Gradient of the entropy term of ELBO wrt to the posterior mean for component ``k`` and latent process ``j``.

        Returns
        -------
        d ent \\ dm[k,j]. Dimensions: M * 1
        """
        m_k = np.zeros(self.num_inducing)
        for l in xrange(self.num_components):
            m_k += (self.gaussian_mixture.weights[k] * self.gaussian_mixture.weights[l] *
                    (np.exp(self.log_N_kl[k, l] - self.log_z[k]) + np.exp(self.log_N_kl[k, l] -
                    self.log_z[l])) * (self.gaussian_mixture.C_m(j, k, l)))
        return m_k

    def _grad_entropy_over_weights(self):
        pi = np.empty(self.num_components)
        for k in range(self.num_components):
            pi[k] = -self.log_z[k]
            for l in range(self.num_components):
                pi[k] -= self.gaussian_mixture.weights[l] * (np.exp(self.log_N_kl[k, l] - self.log_z[l]))
        return pi

    def _d_ent_d_S_kj(self, k, j):
        """
        Calculates gradient of the entropy term of ELBO wrt to the posterior covariance for component ``k`` and latent
        process ``j``. The returned gradient will be in the raw space.
        """
        s_k = np.zeros(self.gaussian_mixture.get_covar_shape())
        for l in range(self.num_components):
            s_k += self.gaussian_mixture.weights[k] * self.gaussian_mixture.weights[l] * (np.exp(self.log_N_kl[l, k] - self.log_z[k]) +
                                                                                np.exp(self.log_N_kl[l, k] - self.log_z[l])) * \
                   self.gaussian_mixture.C_m_C(j, k, l)
        return 1. / 2 * s_k

    def _d_ent_d_S(self):
        r"""
        Calculated gradient of the entropy term of ELBO wrt to the posterior covariance.

        Returns
        -------
        ds : ndarray
         dent \\ ds. Gradients will be in the raw space. Dimensions : K * Q * ``self.MoG.S_dim()``

        """
        dent_ds = np.empty([self.num_components, self.num_latent] + self.gaussian_mixture.get_covar_shape())
        for k in range(self.num_components):
            for j in range(self.num_latent):
                dent_ds[k, j] = self._d_ent_d_S_kj(k, j)
        return dent_ds

    def _grad_entropy_over_covars(self):
        return (self._d_ent_d_S()).flatten()

    def _calculate_entropy(self):
        return -np.dot(self.gaussian_mixture.weights, self.log_z)
