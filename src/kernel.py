from GPy.kern import RBF
from GPy.util.linalg import mdot
import numpy as np
from numpy.core.umath_tests import inner1d


class ExtRBF(RBF):
    """
    Extended-RBF, which extends RBF class in order to provide fast methods for calculating gradients wrt to the
    hyper-parameters of the kernel. The base class provides a similar functionality, but that implementation can be
    slow when evaluating for multiple data points.
    """

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
        variance_gradient = inner1d(self.K(X, X2), A) *  1./ self.variance

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
        if X2 is None:
            tmp = tmp + tmp.T
            X2 = X

        #The high-memory numpy way:
        #d =  X[:, None, :] - X2[None, :, :]
        #ret = np.sum(tmp[:,:,None]*d,1)/self.lengthscale**2

        #the lower memory way with a loop
        ret = np.empty(S.shape + (self.input_dim,))
        for q in xrange(self.input_dim):
            ret[:, :, q] = mdot(tmp * (X[:,q][:,None]-X2[:,q][None,:]), D).T * S + mdot(tmp * (X[:,q][:,None]-X2[:,q][None,:]), S.T).T * D.T
        ret /= self.lengthscale**2

        return ret

    def get_gradients_X_AK(self, A, X, X2=None):
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
        if X2 is None:
            tmp = tmp + tmp.T
            X2 = X

        #The high-memory numpy way:
        #d =  X[:, None, :] - X2[None, :, :]
        #ret = np.sum(tmp[:,:,None]*d,1)/self.lengthscale**2

        #the lower memory way with a loop
        ret = np.empty(A.T.shape + (X.shape[1],), dtype=np.float64)
        for q in xrange(self.input_dim):
            ret[:,:,q] = (tmp*(X[:,q][:, None]-X2[:,q][None, :])).T
        ret /= self.lengthscale**2
        return ret