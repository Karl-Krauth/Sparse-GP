import logging
from kernel import ExtRBF
import run_model
from data_transformation import MeanTransformation
from likelihood import UnivariateGaussian
import data_source
import numpy as np

# defining model type. It can be "mix1", "mix2", or "full"
method = 'full'

# number of inducing points
sparsity_factor = 0.5

# loading data
data = data_source.boston_data()[0]

# is is just of name that will be used for the name of folders and files when exporting results
name = 'boston'

# defining the likelihood function
cond_ll = UnivariateGaussian(np.array(1.0))

# defining the kernel
kernels = [ExtRBF(data['train_X'].shape[1], variance=1, lengthscale=np.array((1.,)), ARD=False)]

run_model.run_model(data['train_X'],
                    data['train_Y'],
                    data['test_X'],
                    data['test_Y'],
                    cond_ll,
                    kernels,
                    method,
                    name,
                    data['id'],
                    sparsity_factor,

                    # optimise hyper-parameters (hyp), posterior parameters (mog),
                      # and likelihood parameters (ll)
                      ['hyp', 'mog', 'll'],

                    # Transform data before training
                      MeanTransformation,

                    # place inducting points on training data.
                      # If False, they will be places using clustering
                      True,

                    # level of logging
                      logging.DEBUG,

                    # do not export training data into csv files
                      False,

                    # number of samples used for approximating the likelihood and its gradients
                      num_samples=2000,

                    # add a small latent noise to the kernel for
                      # stability of numerical computations
                      latent_noise=0.001,

                    # for how many iterations each set of parameters will be optimised
                      opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},

                    # total number of global optimisations
                      max_iter=200,

                    # number of threads
                      n_threads=1,

                    # size of each partition of data
                      partition_size=3000)
