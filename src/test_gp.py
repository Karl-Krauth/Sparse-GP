import numpy as np
import gaussian_process as gp

def kernel_func(x1, x2):
    return np.exp(-np.linalg.norm(x1 - x2) / 2)

def likelihood_func(y, f):
    pass
#    return ((np.power(f, y)).dot(np.exp(-f)).dot())

proc = gp.GaussianProcess([kernel_func], likelihood_func, 0.5)

train_in = np.array([[1., 1., 1.],
                     [2., 2., 2.],
                     [3., 3., 3.]])
train_out = np.zeros([3, 3])

proc.fit(train_in, train_out)
