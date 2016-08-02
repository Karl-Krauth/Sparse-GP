import unittest

import numpy as np

import full_gaussian_process

class TestFullGaussianProcess(unittest.TestCase):
    def test_calculate_entropy_single_latent(self):
        gp = full_gaussian_process.FullGaussianProcess(1, np.ones([1, 5]))
        gp.set_optimization_method('mog')
        gp.set_params()