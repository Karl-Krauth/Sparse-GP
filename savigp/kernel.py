from GPy.kern import RBF
from GPy.util.linalg import mdot
import numpy as np
import util
import torch


class ExtRBF(RBF):
    """
    Extended-RBF, which extends RBF class in order to provide fast methods for calculating gradients wrt to the
    hyper-parameters of the kernel. The base class provides a similar functionality, but that implementation can be
    slow when evaluating for multiple data points.
    """
    white_noise = 0.001
    def get_gradients_AK(self, A, X, X2=None):
        r"""
        Assume we have a function Ln of the kernel, which its gradient wrt to the hyper-parameters (H) is as follows:

         dLn\\dH = An * dK(X2, Xn)\\dH

        where An = A[n, :], Xn = X[n, :]. The function then returns a matrix containing dLn_dH for all 'n's.

        Parameters
        ----------
        A : ndarray
         dim(A) = N * M

        X : ndarray
         dim(X) = N * D

        X2: ndarray
         dim(X2) = M * D

        where D is the dimensionality of input.

        Returns
        -------
        dL_dH : ndarray
         dL\\dH, which is a matrix of dimension N * dim(H), where dim(H) is the number of hyper-parameters.

        """
        kernel = self.kernel(X, X2)
        dk_dr = self.grad_kernel_over_dist(X, X2)

        if self.ARD:
            inv_dist = self._inverse_distances(X, X2)
            if X2 is None: X2 = X
            variance_gradient, lengthscale_gradient = self._torch_get_gradients_AK_ARD(
                X, X2, A, kernel, dk_dr.astype(util.PRECISION), inv_dist.astype(util.PRECISION),
                self.lengthscale.astype(util.PRECISION), self.variance[0])
        else:
            variance_gradient = np.core.umath_tests.inner1d(kernel, A) / self.variance
            dL_dr = (self.dK_dr_via_X(X, X2) * A)
            r = self._scaled_dist(X, X2)
            lengthscale_gradient = -np.sum(dL_dr*r, axis=1)/self.lengthscale
            lengthscale_gradient = lengthscale_gradient[:, np.newaxis]
        return np.hstack((variance_gradient[:, np.newaxis], lengthscale_gradient)).astype(util.PRECISION)

    @util.torchify
    def _torch_get_gradients_AK_ARD(self, X, X2, A, kernel, kd_dr, inv_dist, lengthscale, variance):
        tmp = dk_dr * A * inv_dist
        variance_gradient = torch.sum(kernel * A, dim=1) / variance
        lengthscale_gradient = -(torch.sum(
            tmp[:, :, np.newaxis] * (X[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2,
            dim=1) / lengthscale ** 3)

        return variance_gradient, lengthscale_gradient

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1

        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)
        kernel_value = self._torch_kernel(self.variance[0], length_scale.astype(util.PRECISION), points1, points2)
        if points1 is points2:
            kernel_value += self.white_noise * np.eye(kernel_value.shape[0], dtype=util.PRECISION)

        return kernel_value

    @util.torchify
    def _torch_kernel(self, variance, length_scale, points1, points2):
        scaled_points1 = points1 / length_scale
        scaled_points2 = points2 / length_scale
        magnitude_square1 = torch.sum(scaled_points1 ** 2, dim=1)
        magnitudes_square2 = torch.sum(scaled_points2 ** 2, dim=1)
        distances = (magnitude_square1[:, None] - 2 * scaled_points1.mm(scaled_points2.t()) +
                     magnitudes_square2[:, None].t())

        kernel_value = variance * torch.exp(-distances / 2.0)
        return kernel_value

    def grad_kernel_over_dist(self, points1, points2=None):
        if points2 is None:
            points2 = points1
        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)
        grad_kernel = self._torch_grad_kernel_over_dist(
            self.variance[0], length_scale.astype(util.PRECISION), points1, points2)
        return grad_kernel

    @util.torchify
    def _torch_grad_kernel_over_dist(self, variance, length_scale, points1, points2):
        scaled_points1 = points1 / length_scale
        scaled_points2 = points2 / length_scale
        magnitude_square1 = torch.sum(scaled_points1 ** 2, dim=1)
        magnitudes_square2 = torch.sum(scaled_points2 ** 2, dim=1)
        distances = (magnitude_square1[:, None] - 2 * scaled_points1.mm(scaled_points2.t()) +
                     magnitudes_square2[:, None].t())
        distances = torch.clamp(distances, 0, np.inf)
        grad_kernel = -torch.sqrt(distances) * variance * torch.exp(-distances / 2.0)
        return grad_kernel

    def diag_kernel(self, points):
        return (self.variance.astype(util.PRECISION) + self.white_noise) * np.ones(points.shape[0], dtype=util.PRECISION)

    def _inverse_distances(self, points1, points2=None):
        distances = self._distances(points1, points2)
        return 1.0 / np.where(distances != 0., distances, np.inf)

    def _distances(self, points1, points2=None):
        if points2 is None:
            points2 = points1
        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)
        distances = self._torch_distance(length_scale.astype(util.PRECISION), points1, points2)
        return distances

    @util.torchify
    def _torch_distance(self, length_scale, points1, points2):
        scaled_points1 = points1 / length_scale
        scaled_points2 = points2 / length_scale
        magnitude_square1 = torch.sum(scaled_points1 ** 2, dim=1)
        magnitudes_square2 = torch.sum(scaled_points2 ** 2, dim=1)
        distances_sqr = (
            magnitude_square1[:, None] - 2 * scaled_points1.mm(scaled_points2.t()) +
            magnitudes_square2[:, None].t())
        distances = torch.sqrt(torch.clamp(distances_sqr, 0.0, np.inf))
        return distances

    def get_gradients_Kdiag(self, X):
        r"""
        Assume we have a function Ln of the kernel we follows:

         dL_n\\dH = dK(Xn, Xn)\\dH

        where Xn=X[n, :]. Then the function returns a matrix which contains dL_n\\dH for all 'n's.

        Parameters
        ----------
        X : ndarray
         dim(X) = N * D, where D is the dimension of input.

        Returns
        -------
        dL_dH : ndarray
         dL\\dH which is a matrix of dimension N * dim(H), where dim(H) is the number of hyper-parameters.
        """

        variance_gradient = self.Kdiag(X) * 1./self.variance
        return np.hstack((variance_gradient[:, np.newaxis], np.zeros((X.shape[0], self.lengthscale.shape[0])))).astype(util.PRECISION)

    def get_gradients_SKD(self, S, D, X, X2=None):
        r"""
        Assume we have a function Ln, which its gradient wrt to the hyper-parameters (H), is as follows:
         dLn\\dH = S[n, :] *  dK(X,X2)\\dH * D[:, n]

        then this function calculates dLn\\dH for all 'n's.

        Parameters
        ----------
        S : ndarray
            dim(S) = N * M
        D : ndarray
            dim(D) = M * N
        X : ndarray
            dim(X) = M * d, where d is the input dimensionality \n
        X2 : nadrray
            dim(X2) = M * d

        Returns
        -------
        dL_dH : ndarray
         dL\\dH which is a matrix by dimensions N * dim(H), where dim(H) is the number of hyper-parameters.
        """
        kernel = self.kernel(X, X2)
        dk_dr = self.grad_kernel_over_dist(X, X2)

        if self.ARD:
            inv_dist = self._inverse_distances(X, X2)
            if X2 is None:
                X2 = X
            variance_gradient, lengthscale_gradient = self._torch_get_gradients_SKD_ARD(
                S, D, X, X2, kernel, inv_dist, self.lengthscale.astype(util.PRECISION),
                self.variance[0].astype(util.PRECISION), dk_dr)
        else:
            scaled_dist = self._scaled_dist(X, X2)
            variance_gradient = mdot(S, kernel, D) * 1. / self.variance
            lengthscale_gradient = np.diagonal(-mdot(S, (scaled_dist * dk_dr).T, D) / self.lengthscale)[:, np.newaxis]

        return np.hstack((np.diagonal(variance_gradient)[:, np.newaxis], lengthscale_gradient)).astype(util.PRECISION)

    @util.torchify
    def _torch_get_gradients_SKD_ARD(self, S, D, X, X2, kernel, inv_dist, length_scale, variance, dk_dr):
        diff = X[:, np.newaxis, :] - X2[np.newaxis, :, :]
        x_xl3 = ((diff ** 2) * (inv_dist * dk_dr)[:, :, np.newaxis]).transpose(0, 1)

        variance_gradient = S.mm(kernel).mm(D) / variance
        lengthscale_gradient = -(torch.sum(D.t()[:, :, np.newaxis] * S.mm(x_xl3), dim=1) /
                                 length_scale ** 3)

        return variance_gradient, lengthscale_gradient

    def get_gradients_X_SKD(self, S, D, X):
        r"""
        Assume we have a function Ln, which its gradient wrt to the location of X, is as follows:
         dLn\\dX = S[n, :] *  dK(X)\\dX * D[:, n]

        then this function calculates dLn\\dX for all 'n's.

        Parameters
        ----------
        S : ndarray
            dim(S) = N * M
        D : ndarray
            dim(D) = M * N
        X : ndarray
            dim(X) = M * d, where d is the input dimensionality \n

        Returns
        -------
        dL_dH : ndarray
         dL\\dX which is a matrix by dimensions N * d
        """
        X2 = X
        inv_dist = self._inverse_distances(X, X2)
        dk_dr = self.grad_kernel_over_dist(X, X2)

        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)

        ret = self._torch_get_gradients_X_SKD(X, X2, S, D, inv_dist, dk_dr,
                                              length_scale.astype(util.PRECISION))

        return ret

    @util.torchify
    def _torch_get_gradients_X_SKD(self, X, X2, S, D, inv_dist, dk_dr, length_scale):
        tmp = inv_dist * dk_dr
        tmp2 = tmp[None, :, :] * (X.t()[:, :, None] - X2.t()[:, None, :])
        ret = (tmp2.matmul(D).transpose(1, 2) * S[None, :, :] +
               tmp2.matmul(S.t()).transpose(1, 2) * D.t()[None, :, :])
        # TODO: this might be wrong. need to test this.
        ret = ret.transpose(0, 1).transpose(1, 2)
        ret /= length_scale ** 2
        return ret

    def get_gradients_X_AK(self, A, X, X2):
        r"""
        Assume we have a function Ln of the kernel, which its gradient wrt to the location of X is as follows:

         dLn\\dX = An * dK(X2, Xn)\\dX

        where An = A[n, :], Xn = X[n, :]. The function then returns a matrix containing dLn_dX for all 'n's.

        Parameters
        ----------
        A : ndarray
         dim(A) = N * M

        X : ndarray
         dim(X) = N * D

        X2: ndarray
         dim(X2) = M * D

        where D is the dimensionality of input.

        Returns
        -------
        dL_dX : ndarray
         dL\\dX, which is a matrix of dimension N * D

        """
        inv_dist = self._inverse_distances(X, X2)
        dk_dr = self.grad_kernel_over_dist(X, X2)

        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)

        ret = self._torch_get_gradients_X_AK(X, X2, inv_dist, dk_dr, A,
                                              length_scale.astype(util.PRECISION))

        return ret

    @util.torchify
    def _torch_get_gradients_X_AK(self, X, X2, inv_dist, dk_dr, A, length_scale):
        tmp = inv_dist * dk_dr * A
        ret = tmp.t()[:, :, None] * (X[None, :, :] - X2[:, None, :])
        ret /= length_scale ** 2
        return ret
