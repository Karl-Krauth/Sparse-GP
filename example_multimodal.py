"""
Example of a multimodal posterior
"""


import numpy as np
from savigp.kernel import ExtRBF
from savigp.likelihood import UnivariateGaussian
from savigp.full_gaussian_process import FullGaussianProcess
from savigp.diagonal_gaussian_process import DiagonalGaussianProcess
from savigp.optimizer import batch_optimize_model
from savigp import model_logging
model_logging.init_logger('multimodal_expt', False)

# various parameters
latent_noise = 1e-7
num_samples = 2000
max_iterations = 100
mog_threshold = 1e-3
objective_threshold = 1e-2
n_samples_posterior = 50
posterior = "full"   # full or diag
num_components = 2

# Load Data files generated in deep GP project
X = np.loadtxt("experiments/data/multimodal/X.txt")
Y = np.loadtxt("experiments/data/multimodal/Y.txt")
Xtest = np.loadtxt("experiments/data/multimodal/Xtest.txt")

# just making sure X has shape[1]
X = np.expand_dims(X, axis=1)
Xtest = np.expand_dims(Xtest, axis=1)
Y = np.expand_dims(Y, axis=1)

# Define a univariate Gaussian likelihood function with a variance of 1.
likelihood = UnivariateGaussian(np.array([1.0]))

# Define a radial basis kernel with a variance of 1, lengthscale of 1 and ARD disabled.
kernel = [ExtRBF(1, variance=1.0, lengthscale=np.array([1.0]), ARD=False)]

# Set the number of inducing points to be the whole the training data.
num_inducing = int(X.shape[0])


# Initialize the model.
if posterior == "full":
    gp = FullGaussianProcess(X, Y, num_inducing, num_samples, kernel, likelihood,
                             latent_noise=latent_noise, exact_ell=False, partition_size=X.shape[0])
elif posterior == "diag":
    gp = DiagonalGaussianProcess(X, Y, num_inducing, num_components, num_samples,
                                 kernel, likelihood, latent_noise=latent_noise, exact_ell=False,
                                 partition_size=X.shape[0])
else:
    assert False

# Now fit the model to our training data.
optimization_config = {'mog': 50, 'hyp': 15, 'll': 15}

batch_optimize_model(gp, optimization_config, max_iterations=100,
                     mog_threshold=1e-3, objective_threshold=1e-2)

# Make predictions. The third output is NLPD which is set to None unless test outputs are also
# provided.
predicted_mean, predicted_var, _ = gp.predict(Xtest)

samples_posterior = gp.get_samples_posterior(Xtest, num_samples=n_samples_posterior)

np.savetxt("experiments/data/multimodal/pred_mean.txt", predicted_mean, fmt="%f", delimiter='\t')
np.savetxt("experiments/data/multimodal/pred_var.txt", predicted_var, fmt="%f", delimiter='\t')
np.savetxt("experiments/data/multimodal/post_samples.txt", samples_posterior, fmt="%f", delimiter='\t')

