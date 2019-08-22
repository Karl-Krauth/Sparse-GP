from GPy.kern import RBF
from GPy.util.linalg import mdot
import numpy as np
import theano
import theano.tensor as tensor


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
            variance_gradient, lengthscale_gradient = self._theano_get_gradients_AK_ARD(
                X, X2, A, kernel, dk_dr.astype(np.float32), inv_dist.astype(np.float32),
                self.lengthscale.astype(np.float32), self.variance[0])
        else:
            variance_gradient = np.core.umath_tests.inner1d(kernel, A) / self.variance
            dL_dr = (self.dK_dr_via_X(X, X2) * A)
            r = self._scaled_dist(X, X2)
            lengthscale_gradient = -np.sum(dL_dr*r, axis=1)/self.lengthscale
            lengthscale_gradient = lengthscale_gradient[:, np.newaxis]
        return np.hstack((variance_gradient[:, np.newaxis], lengthscale_gradient)).astype(np.float32)

    def _compile_get_gradients_AK_ARD():
        X = tensor.matrix('X')
        X2 = tensor.matrix('X2')
        A = tensor.matrix('A')
        kernel = tensor.matrix('kernel')
        dk_dr = tensor.matrix('dk_dr')
        inv_dist = tensor.matrix('inv_dist')
        lengthscale = tensor.vector('lengthscale')
        variance = tensor.scalar('variance')

        tmp = dk_dr * A * inv_dist
        variance_gradient = tensor.sum(kernel * A, axis=1) / variance
        lengthscale_gradient = -(tensor.sum(
            tmp[:, :, np.newaxis] * tensor.square(X[:, np.newaxis, :] - X2[np.newaxis, :, :]),
            axis=1) / lengthscale ** 3)

        return theano.function([X, X2, A, kernel, dk_dr, inv_dist, lengthscale, variance],
                               [variance_gradient, lengthscale_gradient])
    _theano_get_gradients_AK_ARD = _compile_get_gradients_AK_ARD()

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1

        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)
        kernel_value = self._theano_kernel(self.variance[0], length_scale.astype(np.float32), points1, points2)
        if points1 is points2:
            kernel_value += self.white_noise * np.eye(kernel_value.shape[0], dtype=np.float32)

        return kernel_value

    def _compile_kernel():
        variance = tensor.scalar('variance')
        length_scale = tensor.vector('length_scale')
        points1 = tensor.matrix('points1')
        points2 = tensor.matrix('points2')

        scaled_points1 = points1 / length_scale
        scaled_points2 = points2 / length_scale
        magnitude_square1 = tensor.sum(tensor.sqr(scaled_points1), 1)
        magnitudes_square2 = tensor.sum(tensor.sqr(scaled_points2), 1)
        distances = (magnitude_square1[:, None] - 2 * tensor.dot(scaled_points1, scaled_points2.T) +
                     magnitudes_square2.T)

        kernel_value = variance * tensor.exp(-distances / 2.0)
        return theano.function([variance, length_scale, points1, points2], kernel_value)
    _theano_kernel = _compile_kernel()

    def grad_kernel_over_dist(self, points1, points2=None):
        if points2 is None:
            points2 = points1
        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)
        grad_kernel = self._theano_grad_kernel_over_dist(
            self.variance[0], length_scale.astype(np.float32), points1, points2)
        return grad_kernel

    def _compile_grad_kernel_over_dist():
        variance = tensor.scalar('variance')
        length_scale = tensor.vector('length_scale')
        points1 = tensor.matrix('points1')
        points2 = tensor.matrix('points2')

        scaled_points1 = points1 / length_scale
        scaled_points2 = points2 / length_scale
        magnitude_square1 = tensor.sum(tensor.sqr(scaled_points1), 1)
        magnitudes_square2 = tensor.sum(tensor.sqr(scaled_points2), 1)
        distances = (magnitude_square1[:, None] - 2 * tensor.dot(scaled_points1, scaled_points2.T) +
                     magnitudes_square2.T)
        distances = tensor.clip(distances, 0, np.inf)

        grad_kernel = -tensor.sqrt(distances) * variance * tensor.exp(-distances / 2.0)
        return theano.function([variance, length_scale, points1, points2], grad_kernel)
    _theano_grad_kernel_over_dist = _compile_grad_kernel_over_dist()

    def diag_kernel(self, points):
        return (self.variance.astype(np.float32) + self.white_noise) * np.ones(points.shape[0], dtype=np.float32)

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
        distances = self._theano_distance(length_scale.astype(np.float32), points1, points2)
        return distances

    def _compile_distance():
        length_scale = tensor.vector('length_scale')
        points1 = tensor.matrix('points1')
        points2 = tensor.matrix('points2')

        scaled_points1 = points1 / length_scale
        scaled_points2 = points2 / length_scale
        magnitude_square1 = tensor.sum(tensor.sqr(scaled_points1), 1)
        magnitudes_square2 = tensor.sum(tensor.sqr(scaled_points2), 1)
        distances_sqr = (
            magnitude_square1[:, None] - 2 * tensor.dot(scaled_points1, scaled_points2.T) +
            magnitudes_square2.T)
        distances = tensor.sqrt(tensor.clip(distances_sqr, 0.0, np.inf))

        return theano.function([length_scale, points1, points2], distances)
    _theano_distance = _compile_distance()

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
        return np.hstack((variance_gradient[:, np.newaxis], np.zeros((X.shape[0], self.lengthscale.shape[0])))).astype(np.float32)

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
            variance_gradient, lengthscale_gradient = self._theano_get_gradients_SKD_ARD(
                S, D, X, X2, kernel, inv_dist, self.lengthscale.astype(np.float32),
                self.variance[0].astype(np.float32), dk_dr)
        else:
            scaled_dist = self._scaled_dist(X, X2)
            variance_gradient = mdot(S, kernel, D) * 1. / self.variance
            lengthscale_gradient = np.diagonal(-mdot(S, (scaled_dist * dk_dr).T, D) / self.lengthscale)[:, np.newaxis]

        return np.hstack((np.diagonal(variance_gradient)[:, np.newaxis], lengthscale_gradient)).astype(np.float32)

    def _compile_get_gradients_SKD_ARD():
        S = tensor.matrix('S')
        D = tensor.matrix('D')
        X = tensor.matrix('X')
        X2 = tensor.matrix('X2')
        kernel = tensor.matrix('kernel')
        inv_dist = tensor.matrix('inv_dist')
        length_scale = tensor.vector('length_scale')
        variance = tensor.scalar('variance')
        dk_dr = tensor.matrix('dk_dr')

        diff = X[:, np.newaxis, :] - X2[np.newaxis, :, :]
        x_xl3 = (tensor.sqr(diff) * (inv_dist * dk_dr)[:, :, np.newaxis]).swapaxes(0, 1)

        variance_gradient = tensor.dot(tensor.dot(S, kernel), D) / variance
        lengthscale_gradient = -(tensor.sum(D.T[:, :, np.newaxis] * tensor.dot(S, x_xl3), axis=1) /
                                 length_scale ** 3)

        return theano.function([S, D, X, X2, kernel, inv_dist, length_scale, variance, dk_dr],
                               [variance_gradient, lengthscale_gradient])
    _theano_get_gradients_SKD_ARD = _compile_get_gradients_SKD_ARD()

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

        ret = self._theano_get_gradients_X_SKD(X, X2, S, D, inv_dist, dk_dr,
                                                length_scale.astype(np.float32))

        return ret

    def _compile_get_gradients_X_SKD():
        X = tensor.matrix('X')
        X2 = tensor.matrix('X2')
        S = tensor.matrix('S')
        D = tensor.matrix('D')
        inv_dist = tensor.matrix('inv_dist')
        dk_dr = tensor.matrix('dk_dr')
        length_scale = tensor.vector('length_scale')

        tmp = inv_dist * dk_dr
        tmp2 = tmp[None, :, :] * (X.T[:, :, None] - X2.T[:, None, :])
        ret = (theano.dot(tmp2, D).swapaxes(1, 2) * S[None, :, :] +
               theano.dot(tmp2, S.T).swapaxes(1, 2) * D.T[None, :, :])
        ret = ret.T.swapaxes(0, 1)
        ret /= length_scale ** 2

        return theano.function([X, X2, S, D, inv_dist, dk_dr, length_scale], ret)

    _theano_get_gradients_X_SKD = _compile_get_gradients_X_SKD()

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

        ret = self._theano_get_gradients_X_AK(X, X2, inv_dist, dk_dr, A,
                                              length_scale.astype(np.float32))

        return ret

    def _compile_get_gradients_X_AK():
        X = tensor.matrix('X')
        X2 = tensor.matrix('X2')
        inv_dist = tensor.matrix('inv_dist')
        dk_dr = tensor.matrix('dk_dr')
        A = tensor.matrix('A')
        length_scale = tensor.vector('lengthscale')

        tmp = inv_dist * dk_dr * A
        ret = tmp.T[:, :, None] * (X[None, :, :] - X2[:, None, :])
        ret /= length_scale ** 2

        return theano.function([X, X2, inv_dist, dk_dr, A, length_scale], ret)
    _theano_get_gradients_X_AK = _compile_get_gradients_X_AK()
