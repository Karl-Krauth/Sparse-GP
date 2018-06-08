__author__ = 'AT'

import math

from GPy.util.linalg import mdot
from numpy.ma import argsort, sort
from scipy.linalg import inv, det
from scipy.misc import logsumexp
from scipy.special import erfinv
#from scipy.special._ufuncs import gammaln
from scipy.special import gammaln
import numpy as np

import theano
from theano import tensor
from util import cross_ent_normal


class Likelihood:
    """
    A generic class which provides likelihood for the model.
    """
    def __init__(self):
        pass

    def ll_F_Y(self, F, Y):
        r"""
        Method which calculates p(Y|F), and dp(Y|F)\\dlambda, for multiple latent function values (``F``), and multiple
        output points (``Y``).

        Parameters
        ----------
        F : ndarray
         dim(F) = S * N * Q,
         where S is the number of samples, N number of datapoints, and Q number of latent processes

        Y : ndarray
         dim(Y) = N * dim(O),
         where dim(O) is the output dimensionality.


        Returns
        -------
        P : ndarray
         dim(P) = N * S
         P[n,s] = p(Y[n,:]| F[s,n,:])

        dP : ndarray
         dim(dP) = N * S
         dP[n,s] = p(Y[n,:]| F[s,n,:])\\dlambda,
         where lambda is the likelihood parameters
        """
        raise Exception("not implemented yet")

    def get_num_params(self):
        """
        :returns number of likelihood parameters to optimize
        """
        raise Exception("not implemented yet")

    def set_params(self, p):
        """
        sets likelihood parameters.

        Parameters
        ----------
        p : ndarray
         array containing values.

        """
        raise Exception("not implemented yet")

    def get_params(self):
        """
        returns likelihood parameters.

        Returns
        -------
        p : ndarray
         parameters of the likelihood.

        """

        raise Exception("not implemented yet")

    def map_Y_to_f(self, Y):
        """
        This functions is used by the model to initialise latent processes.

        Parameters
        ----------
        Y : ndarray
         input matrix. dim(Y) = N * dim(out).

        Returns
        -------
        f : array_like
         initial value of ``f`` given ``Y``. dim(f) = Q, where Q is the number of latent processes.
        """
        return np.mean(Y, axis=0)

    def output_dim(self):
        """
        This function returns dimensionality of the output. It is used by the model to create matrices for prediction.
        """
        raise Exception("not implemented yet")

    def nlpd_dim(self):
        """
        This function returns number of NLPD. This can be useful in the case that for example the likelihood
        returns NLPD for each output separately.
        """
        return 1

    def ell(self, mu, sigma, Y):
        """
        The method returns exact expected log likelihood. It is not generally used by the model, but it is used
        by the grad checker to have access to the exact objective function.

        Returns
        -------
        ell : float
         ell = log \integral p(Y|f)N(f|mu, sigma)

        """

        raise Exception("not implemented yet")

    def predict(self, mu, sigma, Ys, model=None):
        """
        Makes predictions about mean, and variance for the output and calculates NLPD based on Ys.

        Parameters
        ----------
        mu : ndarray
         dim(mu) = N * Q, where Q is the number of latent processes.

        sigma : ndarray
         dim(sigma) = N * Q

        Ys : ndarray
         dim(Ys) = N * output dimension


        Returns
        -------
        mean : array_like
         mean = \integral y * P(y|f)N(f|mu, sigma) df dy

        var :
         variance of the prediction

        NLPD:
         NLPD = -log \integral p(Ys|f)N(f|mu, sigma) df

        """

        raise Exception("not implemented yet")


class MultivariateGaussian(Likelihood):
    """
    Implementation of a multi-variate Gaussian likelihood

     log P(y|f) = -0.5 * log det (sigma) - size(sigma)/2 * log (2 * pi) - 0.5 * (f-y)T sigma^-1 (f-y)
    """
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.sigma = sigma
        self.sigma_inv = inv(self.sigma)
        self.const = -1.0 / 2 * np.log(det(self.sigma)) - float(len(self.sigma)) / 2 * np.log(2 * math.pi)

    def ll_F_Y(self, F, Y):
        c = 1.0 / 2 * (mdot((F-Y), self.sigma_inv) * (F-Y)).sum(axis=2)
        return (self.const + -c), None

    def get_sigma(self):
        return self.sigma

    def get_params(self):
        return self.sigma.flatten()

    def get_num_params(self):
        return self.sigma.flatten().shape[0]

    def ell(self, mu, sigma, Y):
        return cross_ent_normal(mu, np.diag(sigma), Y, self.sigma)

    def output_dim(self):
        return self.sigma.shape[0]


class UnivariateGaussian(Likelihood):
    """
    Implementation of the a univariate likelihood

     log p(y|f) = -0.5 * log(sigma) - 0.5 log (2pi) - 0.5 * (f-y)^2 / sigma
    """
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.set_params(np.log([sigma]))

    def ll_F_Y(self, F, Y):
        c = 1.0 / 2 * np.square(F - Y) / self.sigma
        return (self.const + -c)[:, :, 0], (self.const_grad * self.sigma + c)[:, :, 0]

    def set_params(self, p):
        self.sigma = math.exp(p[0])
        self.const = -1.0 / 2 * np.log(self.sigma) - 1.0 / 2 * np.log(2 * math.pi)
        self.const_grad = -1.0 / 2 / self.sigma

    def get_sigma(self):
        return np.array([[self.sigma]])

    def get_params(self):
        return np.array(np.log([self.sigma]))

    def get_num_params(self):
        return 1

    def predict(self, mu, sigma, Ys, model=None):
        var = sigma + self.sigma
        lpd = None
        if Ys is not None:
            lpd = -(0.5 * np.square(Ys - mu) / var + 0.5 * np.log(2. * math.pi * var))[:, 0]
        return mu, var, lpd[:, np.newaxis] if lpd is not None else None

    def ell(self, mu, sigma, Y):
        return cross_ent_normal(mu, np.diag(sigma), Y, np.array([[self.sigma]]))

    def output_dim(self):
        return 1


class LogGaussianCox(Likelihood):
    """
    Implementation of a Log Gaussian Cox process

     p(y|f) = (lambda)^y exp(-lambda) / y!

     lambda = f + offset
    """

    def __init__(self, offset):
        Likelihood.__init__(self)
        self.offset = offset

    def ll_F_Y(self, F, Y):
        _log_lambda = (F + self.offset)
        return (Y * _log_lambda - np.exp(_log_lambda) - gammaln(Y + 1))[:, :, 0], (Y - np.exp(F + self.offset))[:, :, 0]

    def set_params(self, p):
        self.offset = p[0]

    def get_params(self):
        return np.array([self.offset])

    def get_num_params(self):
        return 1

    def predict(self, mu, sigma, Ys, model=None):
        meanval = np.exp(mu + sigma / 2) * np.exp(self.offset)
        varval = (np.exp(sigma) - 1) * np.exp(2 * mu + sigma) * np.exp(2 * self.offset)
        return meanval, varval, None

    def output_dim(self):
        return 1


class LogisticLL(object, Likelihood):
    """
    Logistic likelihood

     p(y|f) = 1 / (1 + exp(-f))


    The output is assumed to be either 1 or -1

    """

    def __init__(self):
        Likelihood.__init__(self)
        self.n_samples = 20000
        self.normal_samples = np.random.normal(0, 1, self.n_samples).reshape((1, self.n_samples))

    def ll_F_Y(self, F, Y):
        result = -np.log(1 + np.exp(F * Y))[:, :, 0]
        inf_indices = np.where(np.isinf(np.abs(result)))
        if inf_indices[0].size > 0:
            # Deal with the case where exp(f * y) is too large.
            result[inf_indices] = -(F * Y)[:, :, 0][inf_indices]

        return result, None

    def set_params(self, p):
        if p.shape[0] != 0:
            raise Exception("Logistic function does not have free parameters")

    def predict(self, mu, sigma, Ys, model=None):
        f = self.normal_samples * np.sqrt(sigma) + mu
        mean = np.exp(self.ll_F_Y(f.T[:, :, np.newaxis], np.array([[1]]))[0]).mean(axis=0)[:, np.newaxis]
        lpd = None
        if not (Ys is None):
            lpd = np.log((-Ys + 1) / 2 + Ys * mean)

        return mean, mean * (1 - mean), lpd[:, 0][:, np.newaxis]

    def get_params(self):
        return np.array([])

    def get_num_params(self):
        return 0

    def output_dim(self):
        return 1


class SoftmaxLL(Likelihood):
    """
    Softmax likelihood:

    p(y=c|f) = exp(f_c) / (exp(f_1) + ... + exp(f_N))

    output is supposed to be in the form of for example [1 0 0] for class1, and [0 1 0] for class2 etc.
    """

    def __init__(self, dim):
        Likelihood.__init__(self)
        self.dim = dim
        self.n_samples = 20000
        self.normal_samples = np.random.normal(0, 1, self.n_samples * dim) \
            .reshape((self.dim, self.n_samples))

    def ll_F_Y(self, F, Y):
        return self._theano_ll_F_Y(F, Y), None

    def _compile_ll_F_Y():
        F = tensor.tensor3('F')
        Y = tensor.matrix('Y')
        inner_val = F - tensor.sum(F * Y, 2)[:, :, np.newaxis]
        max_val = tensor.max(inner_val, 2)
        result = -(tensor.log(tensor.sum(tensor.exp(inner_val - max_val[:, :, np.newaxis]), 2)) + max_val)
        return theano.function([F, Y], result, allow_input_downcast=True)
    _theano_ll_F_Y = _compile_ll_F_Y()

    def predict(self, mu, sigma, Ys, model=None):
        F = np.empty((self.n_samples, mu.shape[0], self.dim))
        for j in range(self.dim):
            F[:, :, j] = np.outer(self.normal_samples[j, :], np.sqrt(sigma[:, j])) + mu[:, j]
        expF = np.exp(F)
        mean = (expF / expF.sum(2)[:, :, np.newaxis]).mean(axis=0)
        lpd = None
        if not (Ys is None):
            lpd = np.log((Ys * mean).sum(axis=1))
        return mean, None, lpd[:, np.newaxis]


    def set_params(self, p):
        if p.shape[0] != 0:
            raise Exception("Softmax function does not have free parameters")

    def get_params(self):
        return np.array([])

    def get_num_params(self):
        return 0

    def output_dim(self):
        return self.dim


class WarpLL(object, Likelihood):
    """
    Implementation of a Warp likelihood.

    The log likelihood for warped Gaussian processes and its derivatives.
     p(y|f) = dt(y)/dy p(t(y)|f)
     where t(y) = nnwarp(y)

    The likelihood parameters are
     hyp.lik = [a, b ,c log(sqrt(sn2))]
     where a,b,c are parameter vectors of the warping t(y).

    """
    def __init__(self, ea, eb, c, log_s):
        Likelihood.__init__(self)
        self.set_params(np.hstack((ea, eb, c, [log_s])))

    def warp(self, Y):
        ea = np.exp(self.params[0, :])
        eb = np.exp(self.params[1, :])
        c = self.params[2, :]
        tanhcb = np.tanh(np.add.outer(Y, c) * eb)
        t = (tanhcb * ea).sum(axis=2) + Y
        w = ((1. - np.square(tanhcb)) * ea * eb).sum(axis=2) + 1
        return t, w

    def warpinv(self, z, t0, N):
        for n in range(N):
            t1, dt1 = self.warp(t0)
            t0 -= (t1 - z) / dt1
        return t0

    def ll_F_Y(self, F, Y):
        t, w = self.warp(Y)
        sq = 1.0 / 2 * np.square(F - t) / self.sigma
        return (self.const + -sq + np.log(w))[:, :, 0], \
               (self.const_grad * self.sigma + sq)[:, :, 0]

    def set_params(self, p):
        self.sigma = np.exp(p[-1])
        self.const = -1.0 / 2 * np.log(self.sigma) - 1.0 / 2 * np.log(2 * math.pi)
        self.const_grad = -1.0 / 2 / self.sigma
        if p.shape[0] > 1:
            n = (p.shape[0] - 1) / 3
            self.params = p[:-1].reshape(3, n)

    def predict(self, mu, sigma, Ys, model=None):
        #calculating var
        s = sigma + self.sigma
        alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8000, 0.9000])
        q = np.outer(np.sqrt(2 * s), erfinv(2 * alpha - 1)) + mu

        z = self.warp(model.Y)[0]
        I = argsort(z, axis=0)
        sortz = sort(z, axis=0)
        sortt = model.Y[I]

        quant = self.warpinv(q, self._get_initial_points(q, sortz, sortt), 100)
        var = np.square((quant[:, 8] - (quant[:, 0])) / 4)

        #calculating mu
        H = np.array([7.6e-07, 0.0013436, 0.0338744, 0.2401386, 0.6108626, 0.6108626, 0.2401386, 0.0338744, 0.0013436, 7.6e-07])
        quard = np.array([-3.4361591, -2.5327317, -1.7566836, -1.0366108, -0.3429013, 0.3429013, 1.0366108, 1.7566836, 2.5327317, 3.4361591])
        mu_quad = np.outer(np.sqrt(2 * s), quard) + mu
        mean = self.warpinv(mu_quad, self._get_initial_points(mu_quad, sortz, sortt), 100)
        mean = mdot(mean, H[:, np.newaxis]) / np.sqrt(math.pi)
        lpd = None
        if not (Ys is None):
            ts, w = self.warp(Ys)
            lpd = -0.5*np.log(2*math.pi*s) - 0.5 * np.square(ts-mu)/s + np.log(w)
        return mean, var[:, np.newaxis], lpd[:, 0][:, np.newaxis]

    def output_dim(self):
        return 1


    def _get_initial_points(self, q, sortz, sortt):
        t0 = np.empty(q.shape)
        for j in range(q.shape[0]):
            for k in range(q.shape[1]):
                if q[j, k] > sortz[-1]:
                    t0[j,k] = sortt[-1]
                elif q[j,k] < sortz[0]:
                    t0[j,k] = sortt[0]
                else:
                    I = np.argmax(sortz > q[j,k])
                    I = np.array([I - 1, I])
                    t0[j,k] = sortt[I].mean()
        return t0

    def test(self):
        """
        It's a function for testing this class against the Matlob code from which this class is adapted
        """

        mu = np.array([[1.13395340993645e-06, 5.65190424705805e-06, 5.78826209038103e-06, 2.83243484612040e-06, -7.38434570563690e-07]]).T
        sigma = np.array([[ 0.299216202282485, 0.243742513817980, 0.295996476326654, 0.230752860541760, 0.281672812756221
        ]]).T
        Ys = np.array([[-0.200000000000000, -0.150000000000000, -0.100000000000000, -0.150000000000000, -0.250000000000000]]).T
        self.set_params(np.array([-2.0485, 1.7991, 1.5814, 2.7421, 0.9426, 1.7804, 0.1856, 0.7024, -0.7421, -0.0712]))
        self.sigma = 0.8672

    def get_params(self):
        return np.array(np.log([self.sigma]))

    def get_num_params(self):
        return 1


class CogLL(Likelihood):
    """
    Implementation of a Gaussian process network likelihood.

     y ~ N (W  * F, sigma)

     where dim(W) = P * Q, and dim(F) = Q * 1.

    W and F are made of latent processes
    """
    def __init__(self, sigma_y, P, Q):
        """
        :param sigma_y: input noise
        :param P: output dimensionality
        :param Q: number of latent functions in the network
        :return: None
        """
        Likelihood.__init__(self)
        self.P = P
        self.Q = Q
        self.f_num = (P + 1) * Q
        self.set_params(np.array([np.log(sigma_y)]))
        self.n_samples = 20000
        self.normal_samples = np.random.normal(0, 1, self.n_samples * self.f_num) \
            .reshape((self.f_num, self.n_samples))

    def ll_F_Y(self, F, Y):
        W = F[:, :, :self.P * self.Q].reshape(F.shape[0], F.shape[1], self.P, self.Q)
        f = F[:, :, self.P * self.Q:]
        Wf = np.einsum('ijlk,ijk->ijl', W, f)
        c = self._theano_ll_F_Y(Y, Wf, self.sigma_inv)
        return (self.const  -c), (self.const_grad * self.sigma_y + c)

    def _compile_ll_F_Y():
        Y = tensor.matrix('Y')
        Wf = tensor.tensor3('Wf')
        sigma_inv = tensor.matrix('sigma_inv')
        c = 1.0 / 2 * (theano.dot((Y - Wf), sigma_inv) * (Y - Wf)).sum(axis=2)
        return theano.function([Y, Wf, sigma_inv], c)
    _theano_ll_F_Y = _compile_ll_F_Y()

    def get_params(self):
        return np.array([np.log(self.sigma_y)])

    def get_num_params(self):
        return 1

    def ell(self, mu, sigma, Y):
        return cross_ent_normal(mu, np.diag(sigma), Y, self.sigma)

    def output_dim(self):
        return self.P

    def map_Y_to_f(self, Y):
        return np.mean(Y) * np.ones(self.f_num)

    def set_params(self, p):
        self.sigma_y = math.exp(p[0])
        self.sigma = self.sigma_y * np.eye(self.P)
        self.sigma_inv = inv(self.sigma).astype(np.float32)
        self.const = -1.0 / 2 * np.log(det(self.sigma)) - float(len(self.sigma)) / 2 * np.log(2 * math.pi)
        self.const_grad = -float(self.P) / 2. / self.sigma_y

    def predict(self, mu, sigma, Ys, model=None):
        F = np.empty((self.n_samples, mu.shape[0], self.f_num))
        for j in range(self.f_num):
            F[:, :, j] = np.outer(self.normal_samples[j, :], np.sqrt(sigma[:, j])) + mu[:, j]

        W = F[:, :, :self.P * self.Q].reshape(F.shape[0], F.shape[1], self.P, self.Q)
        f = F[:, :, self.P * self.Q:]
        Wf = np.einsum('ijlk,ijk->ijl', W, f)
        lpd = None
        if Ys is not None:
            lpd = self._calc_nlpd(Ys, Wf)
        return Wf.mean(axis=0), None, lpd

    def _calc_nlpd(self, Ys, Wf):
        lpd = np.empty((Ys.shape[0], Ys.shape[1] + 1))
        c = 1.0 / 2 * (mdot((Ys - Wf), self.sigma_inv) * (Ys - Wf)).sum(axis=2)
        lpd[:, 0] = np.log(np.exp(self.const + -c).mean(axis=0))
        for i in range(Ys.shape[1]):
            c = 1.0 / 2 * (np.square((Ys[:, i] - Wf[:, :, i])) * self.sigma_inv[i,i])
            const = -1.0 / 2 * np.log((self.sigma[i,i])) - 1. / 2 * np.log(2 * math.pi)
            lpd[:, i+1] = np.log(np.exp(const + -c).mean(axis=0))

        return lpd

    def nlpd_dim(self):
        return self.P + 1


class SeismicLL(Likelihood):

    def __init__(self, P, sigma2y):
        Likelihood.__init__(self)
        self.P = P
        self.f_num = 2 * P
        self.sigma2y = sigma2y

    def ll_F_Y(self, F, Y):
        depth = F[:, :, 0:self.P]
        vel = F[:, :, self.P:]

        G = np.zeros((F.shape[0], Y.shape[0], Y.shape[1]))

        G[:, :, 0] = 2. * depth[:, :, 0] / vel[:, :, 0]
        for p in range(1, self.P):
            G[:, :, p] = G[:, :, p-1] + 2 * (depth[:, :, p] - depth[:, :, p - 1]) / vel[:, :, p]

        return (- (0.5 * ((G - Y) ** 2) / self.sigma2y) - 0.5 * np.log(2 * np.pi)
                - 0.5 * np.log(self.sigma2y)).sum(axis=2), None

    def get_params(self):
        return np.array(np.log([self.sigma]))

    def get_num_params(self):
        return 1

    def ell(self, mu, sigma, Y):
        pass

    def output_dim(self):
        return 2 * self.P

    def map_Y_to_f(self, Y):
        return np.array([200, 500, 1600, 2200, 1950, 2300, 2750, 3650])
        # Now considering non-zero mean GP
        # return np.array([0, 0, 0, 0, 0, 0, 0, 0])

    def predict(self, mu, sigma, Ys, model=None):
        return mu, sigma, np.zeros((Ys.shape[0], 1))
