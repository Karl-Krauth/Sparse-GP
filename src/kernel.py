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
        variance_gradient = np.core.umath_tests.inner1d(self.K(X, X2), A) *  1./ self.variance

        dL_dr = (self.dK_dr_via_X(X, X2) * A)
        if self.ARD:
            tmp = dL_dr * self._inv_dist(X, X2)
            if X2 is None: X2 = X
            lengthscale_gradient = np.array([np.einsum('ij,ij,...->i', tmp, np.square(X[:,q:q+1] - X2[:,q:q+1].T), -1./self.lengthscale[q]**3)
                                             for q in xrange(self.input_dim)])
        else:
            r = self._scaled_dist(X, X2)
            lengthscale_gradient = -np.sum(dL_dr*r, axis=1)/self.lengthscale
            lengthscale_gradient = lengthscale_gradient[np.newaxis, :]

        return np.hstack((variance_gradient[:, np.newaxis], lengthscale_gradient.T))

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1

        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)
        kernel_value = self._theano_kernel(self.variance[0], length_scale, points1, points2)
        if points1 is points2:
            kernel_value += self.white_noise * np.eye(kernel_value.shape[0])

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
        return theano.function([variance, length_scale, points1, points2], kernel_value,
                               allow_input_downcast=True)
    _theano_kernel = _compile_kernel()

    def diag_kernel(self, points):
        return (self.variance + self.white_noise) * np.ones(points.shape[0])

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
        return np.hstack((variance_gradient[:, np.newaxis], np.zeros((X.shape[0], self.lengthscale.shape[0]))))

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
        variance_gradient = mdot(S, self.K(X, X2), D) * 1./self.variance

        if X2 is None: X2 = X
        if self.ARD:
            rinv = self._inv_dist(X, X2)
            d =  X[:, None, :] - X2[None, :, :]
            x_xl3 = np.square(d) * (rinv * self.dK_dr_via_X(X, X2))[:,:,None]
            lengthscale_gradient = -np.tensordot(D, np.tensordot(S, x_xl3, (1,0)), (0,1)) / self.lengthscale**3
            lengthscale_gradient = np.diagonal(lengthscale_gradient).T
        else:
            lengthscale_gradient = np.diagonal(-mdot(S, (self._scaled_dist(X, X2) * self.dK_dr_via_X(X, X2)).T, D) / self.lengthscale)[:, np.newaxis]

        return np.hstack((np.diagonal(variance_gradient)[:, np.newaxis], lengthscale_gradient))

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
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2)
        tmp = invdist*dL_dr

        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)

        ret = self._theano_get_gradients_X_SKD(tmp, X, X2, S, D, length_scale)

        return ret

    def _compile_get_gradients_X_SKD():
        tmp = tensor.matrix('tmp')
        X = tensor.matrix('X')
        X2 = tensor.matrix('X2')
        S = tensor.matrix('S')
        D = tensor.matrix('D')
        length_scale = tensor.vector('length_scale')

        tmp2 = tmp[None, :, :] * (X.T[:, :, None] - X2.T[:, None, :])
        ret = (theano.dot(tmp2, D).swapaxes(1, 2) * S[None, :, :] +
               theano.dot(tmp2, S.T).swapaxes(1, 2) * D.T[None, :, :])
        ret = ret.T.swapaxes(0, 1)
        ret /= length_scale ** 2

        return theano.function([tmp, X, X2, S, D, length_scale], ret, allow_input_downcast=True)

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
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2) * A
        tmp = invdist*dL_dr

        if self.lengthscale.shape[0] == self.input_dim:
            length_scale = self.lengthscale
        else:
            length_scale = self.lengthscale.repeat(self.input_dim)

        ret = self._theano_get_gradients_X_AK(tmp, X, X2, length_scale)

        return ret

    def _compile_get_gradients_X_AK():
        tmp = tensor.matrix('tmp')
        X = tensor.matrix('X')
        X2 = tensor.matrix('X2')
        length_scale = tensor.vector('lengthscale')

        ret = tmp.T[:, :, None] * (X[None, :, :] - X2[:, None, :])
        ret /= length_scale ** 2

        return theano.function([tmp, X, X2, length_scale], ret, allow_input_downcast=True)
    _theano_get_gradients_X_AK = _compile_get_gradients_X_AK()