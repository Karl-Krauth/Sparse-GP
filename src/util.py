import math
import os
import random
import string
from numpy.core.umath_tests import inner1d
from numpy.ma import trace
from scipy import linalg
from scipy.linalg import det, inv, lapack
from GPy.util.linalg import mdot, dpotri
import numpy as np


class PosDefMatrix(object):
    def __init__(self, num_latent, num_inducing):
        self.is_outdated = True
        self.matrix = np.empty([num_latent, num_inducing, num_inducing])
        self.inverse = np.empty([num_latent, num_inducing, num_inducing])
        self.cholesky = np.empty([num_latent, num_inducing, num_inducing])
        self.log_determinant = np.empty([num_latent])

    def update(self, kernels, inducing_locations):
        if not self.is_outdated:
            return

        for i in xrange(len(kernels)):
            self.matrix[i] = kernels[i].kernel(inducing_locations[i])
            self.cholesky[i] = jitchol(self.matrix[i])
            self.inverse[i] = inv_chol(self.cholesky[i])
            self.log_determinant[i] = pddet(self.cholesky[i])

        self.is_outdated = False

    def set_outdated(self):
        self.is_outdated = True


def weighted_average(weights, points, num_samples):
    """
    calculates (condll * X).mean(axis=1) using variance reduction method.

    number of control variables = number of samples / 10

    Parameters
    ----------
    condll : ndarray
        dimensions: s * N
    X : ndarray
        dimensions: s * N

    Returns
    -------
    :returns: a matrix of dimension N
    """
    points = points.T
    weights = weights.T
    cvsamples = num_samples / 10
    pz = points[:, 0:cvsamples]
    py = np.multiply(weights[:, 0:cvsamples], pz)
    above = np.multiply((py.T - py.mean(1)), pz.T).sum(axis=0) / (cvsamples - 1)
    below = np.square(pz).sum(axis=1) / (cvsamples - 1)
    cvopt = np.divide(above, below)
    cvopt = np.nan_to_num(cvopt)
    grads = np.multiply(weights, points) - np.multiply(cvopt, points.T).T

    return grads.mean(axis=1)

def mdiag_dot(A, B):
    """
    Given input matrices ``A`` and ``B``, this function returns the diagonal terms of the matrix product of A and B

    Returns
    -------
    output : ndarray
     diag(AB)
    """
    return np.einsum('ij,ji -> i', A, B)


def KL_normal(m1, sigma1, m2, sigma2):
    """
    Calculates the KL divergence between two normal distributions specified by
    N(``mu1``, ``sigma1``), N(``mu2``, ``sigma2``)
    """

    return 1. / 2. * (math.log(det(sigma2) / det(sigma1)) - len(m1) + trace(mdot(inv(sigma2), sigma1)) + \
    mdot((m2 - m1).T, inv(sigma2) , m2- m1))


def cross_ent_normal(m1, sigma1, m2, sigma2):
    """
    Calculates the cross entropy between two normal distributions specified by
    N(``mu1``, ``sigma1``), N(``mu2``, ``sigma2``)
    """

    return -KL_normal(m1, sigma1, m2, sigma2) - 1. / 2 * math.log(det(2.0 * math.pi * math.e * sigma1))


def jitchol(A, maxtries=5):
    """
    Calculates the Cholesky decomposition of ``A``. In the case that it is not possible to calculate the Cholesky,
    a jitter will be added to ``A``.

    Note
    ----
    This method is adopted from the GPy package
    """

    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise JitChol, "not pd: non-positive diagonal elements"
        jitter = diagA.mean() * 1e-6
        while maxtries > 0 and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise JitChol, "not positive definite, even with jitter."


def pddet(L):
    """
    Determinant of a positive definite matrix, only symmetric matricies though

    Note
    ----
    This method is adopted from the GPy package
    """

    logdetA = 2*sum(np.log(np.diag(L)))
    return logdetA


def inv_chol(L):
    """
    Given that ``L`` is the Cholesky decomposition of A, this method returns A^-1

    Note
    ----
    This method is adopted from the GPy package
    """

    Ai, _ = dpotri(np.asfortranarray(L), lower=1)
    return Ai


def log_diag_gaussian(m1, m2, s_log):
    """
    Returns PDF of a normal distribution as follows:

    N(m1| m2, exp(s_log)), where the covariance matrix is diagonal
    """
    const = -1.0 / 2 * s_log.sum() - float(len(s_log)) / 2 * np.log(2 * math.pi)
    return const + -1.0 / 2 * np.dot((m1 - m2) / np.exp(s_log), (m1-m2).T)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def check_dir_exists(dir_name):
    """
    Checks if folder ``dir_name`` exists, and if it does not exist, it will be created.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    """generates a random sequence of character of length ``size``"""

    return ''.join(random.choice(chars) for _ in range(size))


def tr_AB(A, B):
    """ Given two matrices ``A`` and ``B``, this function return trace (AB) """
    return np.sum(inner1d(A, B.T))


def get_git():
    """
    If the current directory is a git repository, this function extracts the hash code, and current branch

    Returns
    -------
    hash : string
     hash code of current commit

    branch : string
     current branch
    """
    try:
        from subprocess import Popen, PIPE

        gitproc = Popen(['git', 'show-ref'], stdout = PIPE)
        (stdout, stderr) = gitproc.communicate()

        gitproc = Popen(['git', 'rev-parse',  '--abbrev-ref',  'HEAD'], stdout = PIPE)
        (branch, stderr) = gitproc.communicate()
        branch = branch.split('\n')[0]
        for row in stdout.split('\n'):
            if row.find(branch) != -1:
                hash = row.split()[0]
                break
    except:
        hash = None
        branch = None
    return hash, branch


def drange(start, stop, step):
    """
    Generates an array of floats starting from ``start`` ending with ``stop`` with step ``step``
    """
    r = start
    while r < stop:
        yield r
        r += step


class JitChol(Exception):
    pass
