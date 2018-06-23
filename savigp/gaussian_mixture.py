"""
This module contains the base class representing a mixture of Gaussians.

Support for specific operations related to the posterior distribution used to represent
a gaussian process are given.
We note that a raw parameter is a parameter seen by the optimizer and might not be how it is
represented internally.
"""
from GPy.util.linalg import mdot
import numpy as np
import util


class GaussianMixture(object):
    """
    The base class that represents a generic mixture of Gaussians.

    We expect any subclass to implement a representation for the covariances as well as associated
    methods. Simplified covariances, like a covariance matrix consisting of only diagonals are
    supported.

    Parameters
    ----------
    num_components : int
        The number of components the mixture of Gaussians has.
    num_latent : int
        The number of latent processes the mixture of Gaussians outputs.
    initial_mean : ndarray
        The initial mean of the mixture of Gaussians which will be applied across all components.
        Dimensions: num_latent * num_dim.
    """
    def __init__(self, num_components, num_latent, initial_mean):
        self.num_components = num_components
        self.num_latent = num_latent
        self.num_dim = initial_mean.shape[1]

        self.means = np.empty([self.num_components, self.num_latent, self.num_dim],
                              dtype=util.PRECISION)
        self.weights = np.empty(self.num_components, dtype=util.PRECISION)
        self.unnormalized_weights = np.empty(self.num_components, dtype=util.PRECISION)

        for i in xrange(self.num_components):
            self.means[i] = initial_mean

        self.set_weights(np.random.uniform(low=1.0, high=5.0, size=self.num_components).
                         astype(util.PRECISION))

    def get_params(self):
        """
        Get the raw parameters of the mixture of Gaussians.

        Returns
        -------
        ndarray
            The parameters of the mixture of Gaussians in a single array.
            Dimensions: means_size + covars_size + weights_size.
        """
        raise NotImplementedError

    def get_means_and_covars(self):
        """
        Get the means and covariances in the raw space.

        Returns
        -------
        means : ndarray
            The means of the mixture of Gaussians.
            Dimensions: num_components * num_latent * num_dim.
        covars : ndarray
            The raw covariance of the mixture of Gaussians.
            Dimensions: num_components * num_latent * get_covar_size()
        """
        raise NotImplementedError

    def set_params(self, params):
        """
        Set the parameters of the mixture of Gaussians using raw parameters.

        Parameters
        ----------
        params : ndarray
            The raw parameters representing the new mixture of Gaussian parameters.
            Dimension: means_size + covars_size + weights_size.
        """
        means_size = self.num_components * self.num_latent * self.num_dim
        covars_size = self.num_components * self.num_latent * self.get_covar_size()
        self.set_means(params[:means_size])
        self.set_covars(params[means_size:(means_size + covars_size)])
        self.set_weights(params[(means_size + covars_size):])

    def set_means(self, raw_means):
        """
        Set the means of the mixture of Gaussians using the raw means.

        Parameters
        ----------
        raw_means : ndarray
            The new means represented in raw form. Dimension: means_size.
        """
        self.means = raw_means.reshape([self.num_components, self.num_latent, self.num_dim])

    def set_covars(self, raw_covars):
        """
        Set the covariances of the mixture of Gaussians using the raw covariances.

        Parameters
        ----------
        raw_covars : ndarray
            The new covariances represented in raw form. Dimension: covars_size.
        """
        raise NotImplementedError

    def set_weights(self, raw_weights):
        """
        Set the weights of the mixture of Gaussians using the raw weights.

        Parameters
        ----------
        raw_weights : ndarray
            The new weights represented in raw form. Dimension: num_components.
        """
        self.unnormalized_weights = np.exp(raw_weights)
        self.weights = self.unnormalized_weights / sum(self.unnormalized_weights)

    def transform_covars_grad(self, internal_grad):
        """
        Transform a gradient with respect to the internal covariances into the gradient with
        respect to the raw covariances.

        Parameters
        ----------
        internal_grad : ndarray
            The gradient with respect to the internal covariances.
            Dimensions: num_components * num_latent * get_covar_dim.

        Returns
        -------
        ndarray
            The gradient with respect to the raw covariances. Dimension: covars_size.
        """
        raise NotImplementedError

    def transform_weights_grad(self, internal_grad):
        """
        Transform a gradient with respect to the internal weights into the gradient with
        respect to the raw weights.

        Parameters
        ----------
        internal_grad : ndarray
            The gradient with respect to the internal weights. Dimension: num_components.

        Returns
        -------
        ndarray
            The gradient with respect to the raw weights. Dimension: num_components.
        """
        pit = np.repeat(np.array([self.weights.T], dtype=util.PRECISION), self.num_components, 0)
        dpi_dx = pit * (-pit.T + np.eye(self.num_components, dtype=util.PRECISION))
        return mdot(internal_grad, dpi_dx)

    def get_covar_shape(self):
        """
        Get the shape of a single gaussian mixture covariance matrix for one component and latent
        process.

        Returns
        -------
        list of int
            The shape of a single covariance matrix.
        """
        raise NotImplementedError

    def get_covar_size(self):
        """
        Get the number of array elements needed to represent a single covariance matrix for one
        component and one latent process in the raw space.
        """
        raise NotImplementedError

    def trace_with_covar(self, A, k, j):
        # TODO(karl): comment
        raise NotImplementedError

    def a_dot_covar_dot_a(self, a, k, j):
        """
        Get the diagonal of the quadratic form of the j-th covariance matrix with respect to the
        vector a.

        Parameters
        ----------
        a : ndarray
            The vector with respect to which we are calculating the quadratic form.
        component_index : int
            The index of the mixture of gaussian component we are operating on.
        latent_index : int
            The index of the latent process we are operating on.

        Returns
        -------
        float
            The diagonal of the quadratic form.
        """
        raise NotImplementedError

    def mean_prod_sum_covar(self, k, j):
        # TODO(karl): comment
        raise NotImplementedError

    def grad_trace_a_inv_dot_covars(self, chol_a, k, j):
        # TODO(karl): comment
        raise NotImplementedError

    def covar_dot_a(self, a, k, j):
        # TODO(karl): comment
        raise NotImplementedError
