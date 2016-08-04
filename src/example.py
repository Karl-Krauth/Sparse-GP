import logging
from kernel import ExtRBF
import run_model
import data_transformation
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
kernel = [ExtRBF(data['train_inputs'].shape[1], variance=1, lengthscale=np.array((1.,)), ARD=False)]

# Transform data before training
transform = data_transformation.MeanTransformation(data['train_inputs'], data['train_outputs'])


run_model.run_model(data['train_inputs'],
                    data['train_outputs'],
                    data['test_inputs'],
                    data['test_outputs'],
                    cond_ll,
                    kernel,
                    method,
                    name,
                    data['id'],
                    sparsity_factor,
                    transform,
                    True,                     # place inducting points on training data (If False use clustering)
                    False,                    # Do not export X,
                    optimization_config={'mog': 25, 'hyp': 25, 'll': 25}, # iterations for each parameter
                    )
