import numpy as np

import data_source
import data_transformation
from kernel import ExtRBF
from likelihood import UnivariateGaussian
from savigp import Savigp

# Load the boston dataset.
data = data_source.boston_data()[0]

# Define a univariate Gaussian likelihood function with a variance of 1.
likelihood = UnivariateGaussian(np.array([1.0]))

# Define a radial basis kernel with a variance of 1, lengthscale of 1 and ARD disabled.
kernel = [ExtRBF(data['train_inputs'].shape[1],
                 variance=1.0,
                 lengthscale=np.array([1.0]),
                 ARD=False)]

# Set the number of inducing points to be half of the training data.
num_inducing = int(0.5 * data['train_inputs'].shape[0])

# Transform the data before training.
transform = data_transformation.MeanTransformation(data['train_inputs'], data['train_outputs'])
train_inputs = transform.transform_X(data['train_inputs'])
train_outputs = transform.transform_Y(data['train_outputs'])
test_inputs = transform.transform_X(data['test_inputs'])

# Initialize the model.
gp = Savigp(likelihood=likelihood,
            kernels=kernel,
            num_inducing=num_inducing,
            debug_output=True)

# Now fit the model to our training data.
gp.fit(train_inputs, train_outputs)

# Make predictions. The third output is NLPD which is set to None unless test outputs are also
# provided.
predicted_mean, predicted_var, _ = gp.predict(data['test_inputs'])

# Untransform the results.
predicted_mean = transform.untransform_Y(predicted_mean)
predicted_var = transform.untransform_Y_var(predicted_var)

# Print the mean standardized squared error.
test_outputs = data['test_outputs']
print "MSSE:", (((predicted_mean - test_outputs) ** 2).mean() /
                ((test_outputs.mean() - test_outputs) ** 2).mean())
