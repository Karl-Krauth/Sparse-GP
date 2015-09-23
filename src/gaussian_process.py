import numpy as np
import scipy.linalg as la

class GaussianProcess(object):
    def __init__(self,
                 kernels,
                 likelihood,
                 inducing_density,
                 num_components=10,
                 num_gaussian_samples=40,
                 step_size=0.01,
                 stopping_threshold=0.0001):
        self._kernels = kernels
        self._likelihood = likelihood
        self._inducing_density = inducing_density
        self._num_components = num_components
        self._num_latent = len(self._kernels)
        self._num_gaussian_samples = num_gaussian_samples
        self._step_size = step_size
        self._stopping_threshold = stopping_threshold
        self._num_inducing = None

    def fit(self, training_inputs, training_outputs):
        # Initialize data dimensionality information.
        num_training, num_features = training_inputs.shape
        num_outputs = training_outputs.shape[1]
        self._num_inducing = np.ceil(self._inducing_density * num_training)

        # Choose the inducing inputs by randomly selecting amongst samples.
        # Each latent function has num_inducing points.
        inducing_inputs = np.empty([self._num_latent, self._num_inducing,
                                    num_features])
        for i in xrange(self._num_latent):
            # TODO: This can be sped-up. Also do we really need different
            # inducing points per latent function?
            inducing_inputs[i] = np.random.permutation(
                training_inputs)[:self._num_inducing]

        # Initialize the mixture of gaussian parameters. We have a mixture
        # with num_components. Each component consists of num_latent gaussians
        # each of which is a num_inducing dimensional gaussian.
        # TODO: Maybe we should initialize things to other values?
        # how do we ensure the matrix stays non-singular?
        mog_means = np.ones([self._num_components, self._num_latent,
                              self._num_inducing])
        mog_covars = np.empty([self._num_components, self._num_latent,
                               self._num_inducing, self._num_inducing])
        mog_weights = np.ones([self._num_components])
        mog_proportions = mog_weights / np.sum(mog_weights)
        for i in xrange(self._num_components):
            for j in xrange(self._num_latent):
                mog_covars[i, j] = np.eye(self._num_inducing)

        # Pre-compute the covariance matrix of the inducing points and its
        # inverse and determinant. Since the matrices are block diagonal we
        # store them as num_latent seperate sub-matrices of size num_inducing *
        # num_inducing.
        # TODO: Check inducing covariances are block diagonal.
        inducing_covars = np.empty([self._num_latent, self._num_inducing,
                                    self._num_inducing])
        inducing_covars_inverse = np.empty(inducing_covars.shape)
        det_inducing_covars = np.empty([self._num_latent])
        for i in xrange(self._num_latent):
            inducing_covars[i] = self._generate_covariance_matrix(
                inducing_inputs[i], self._kernels[i])
            # TODO: Can we take advantage of the symmetry of the matrix?
            # How do we deal with singular matrices? 
            inducing_covars_inverse[i] = np.linalg.inv(inducing_covars[i]) 
            det_inducing_covars[i] = np.linalg.det(inducing_covars[i])

        self._mog_gradient_descent(mog_means, mog_covars, mog_weights,
                                   mog_proportions, inducing_covars,
                                   inducing_covars_inverse, det_inducing_covars)

    def predict(self, X):
        pass

    def _mog_gradient_descent(self, mog_means, mog_covars, mog_weights,
                              mog_proportions, inducing_covars,
                              inducing_covars_inverse, det_inducing_covars):
        # Initialize all gradients to zero.
        means_grad = np.zeros(mog_means.shape)
        covars_grad = np.zeros(mog_covars.shape)
        weights_grad = np.zeros(mog_weights.shape)

        # Now calculate the gradients.
        cross_means_grad, cross_covars_grad, cross_weights_grad = (
            self._cross_entropy_grad(mog_means, mog_covars, mog_weights,
                                     mog_proportions, inducing_covars_inverse,
                                     det_inducing_covars))
        means_grad += cross_means_grad
        covars_grad += cross_covars_grad
        weights_grad += cross_weights_grad

        ent_means_grad, ent_covars_grad, ent_weights_grad = (
            self._entropy_grad(mog_means, mog_covars, mog_weights,
                               mog_proportions))

    def _cross_entropy_grad(self, mog_means, mog_covars, mog_weights,
                            mog_proportions, inducing_covars_inverse,
                            det_inducing_covars):
        means_grad = np.zeros(mog_means.shape)
        covars_grad = np.zeros(mog_covars.shape)
        weights_grad = np.zeros(mog_weights.shape)
        proportions_grad = np.zeros(mog_proportions.shape)
        for i in xrange(self._num_components):
            for j in xrange(self._num_latent):
                means_grad[i, j] = (-mog_proportions[i] *
                                    inducing_covars_inverse[j].
                                    dot(mog_means[i, j]))
                covars_grad[i, j] = (-0.5 * mog_proportions[i] *
                                     inducing_covars_inverse[j])
                # TODO: Is the log base e? Consider seperating into functions.
                # Is the trace of both k(Z,Z)^-1 and S_kj or just the first?
                proportions_grad[i] += (
                    -0.5 * (self._num_inducing * np.log(2 * np.pi) +
                    np.log(det_inducing_covars[j]) + mog_means[i, j].
                    dot(inducing_covars_inverse[j]).dot(mog_means[i, j]) +
                    np.trace(inducing_covars_inverse[j].
                    dot(mog_covars[i, j]))))

        weights_grad = self._proportions_to_weights_grad(mog_weights,
                                                         proportions_grad)
        print "============================="
        print "========cross entropy========"
        print "============================="
        print "means grad"
        print means_grad, "\n\n"
        print "covars grad\n", covars_grad, "\n\n"
        print "weights grad\n", weights_grad
        print "proportions grad\n", proportions_grad, "\n\n"
        return (means_grad, covars_grad, weights_grad)

    def _entropy_grad(self, mog_means, mog_covars, mog_weights,
                      mog_proportions):
        # Pre-compute the normals and their weighted sum.
        normals = np.ones([self._num_components, self._num_components])
        weighted_normal_sums = np.empty([self._num_components])
        for i in xrange(self._num_components):
            for j in xrange(self._num_components):
                for k in xrange(self._num_latent):
                    # TODO(karl): We can pre-compute the inverse of sums at the
                    # cost of storage space. Investigate if this is worth doing.
                    # (It'll result in a 4x speedup for this function.)
                    normals[i, j] *= self._gaussian_density(mog_means[i, k],
                                                            mog_means[j, k],
                                                            mog_covars[i, k] +
                                                            mog_covars[j, k])
            weighted_normal_sums[i] = mog_proportions.dot(normals[i])

        # Now calculate the gradients.
        means_grad = np.zeros(mog_means.shape)
        covars_grad = np.zeros(mog_covars.shape)
        proportions_grad = -np.log(weighted_normal_sums)
        for i in xrange(self._num_components):
            for j in xrange(self._num_components):
                # Join the means and covariances for each latent function
                # into one vector and matrix.
                mean_diff = np.concatenate(mog_means[i] - mog_means[j])
                # TODO(karl): Can we speed this up by calculating sum
                # of inverses from their components?
                sum_inv = la.block_diag(*np.linalg.inv(mog_covars[i] +
                                                       mog_covars[j]))
                full_means = (mog_proportions[i] * mog_proportions[j] *
                             (normals[i, j] / weighted_normal_sums[i] +
                             normals[i, j] / weighted_normal_sums[j]) *
                             sum_inv.dot(mean_diff))
                full_covars = (0.5 * mog_proportions[i] * mog_proportions[j] *
                              (normals[i, j] / weighted_normal_sums[i] +
                              normals[i, j] / weighted_normal_sums[j]) *
                              (sum_inv - sum_inv.dot(np.outer(mean_diff,
                              mean_diff)).dot(sum_inv)))
                # Now split up the means and covariances into a vector and
                # matrix for each latent function again.
                means_grad[i] += np.split(full_means, self._num_latent)
                covars_grad[i] += np.split(full_covars, self._num_latent)
                proportions_grad[i] -= (mog_proportions[j] * normals[i, j] /
                                       weighted_normal_sums[j])

        weights_grad = self._proportions_to_weights_grad(mog_weights,
                                                         proportions_grad)

        print "============================="
        print "===========entropy==========="
        print "============================="
        print "ent means grad:"
        print means_grad, "\n\n"
        print "covars grad"
        print covars_grad, "\n\n"
        print "weights grad"
        print weights_grad,
        print "props grad"
        print proportions_grad, "\n\n\n"
        return (means_grad, covars_grad, weights_grad)

    def _ell_grad(self):
        # Calculate intermediate values on the training point.
        train_covars = np.empty([self._num_latent, num_inducing])
        train_products = np.empty([self._num_latent, num_inducing])
        for i in xrange(self._num_latent):
            train_covars[i] = self._generate_covar_vector(training_points,
                                                          inducing_points,
                                                          self._kernels[i])
            train_products[i] = train_covar.dot(inducing_covars_inv[i])
            
        for i in xrange(self._num_components):
            # Calculate the sample gaussian parameters.
            sample_means = np.empty([self._num_latent])
            sample_vars = np.empty([self._num_latent])
            for j in xrange(self._num_latent):
                sample_means[j] = train_products[j].dot(mog_means[i, j])
                sample_vars[j] = ((self._kernels[j](training_point) -
                                 train_products[j].dot(train_covars[j])) +
                                 train_product[j].dot(mog_covar[i, j]).
                                 dot(train_products[j]))

            # Generate the samples.
            samples = np.empty([self._num_samples, self._num_latent])
            for j in xrange(self._num_samples):
                for k in xrange(self._num_latent):
                    samples[j, k] = np.random.normal(sample_means[k],
                                                     sample_vars[k])

            # Now Calculate the gradients.
            for j in xrange(self._num_samples):
                for k in xrange(self._num_latent):
                    means_grad[i, k] += (
                        (mog_proportions[i] / self._num_samples) *
                        (inducing_covars_inv[k].dot(training_covars[k]) /
                        sample_vars[k]) * (samples[j, k] - sample_means[k]) *
                        self._likelihood(train_output, samples[j])
                    covars_grad[i, k] += (
                        (mog_proportions[i] / (2 * self._num_samples)) *
                        np.outer(train_products[k], train_products[k]) *
                        (np.power(sample_vars[k], -2) *
                        np.power((samples[j, k] - sample_means[k]), 2) -
                        1.0 / sample_vars[k]) * self._likelihood(train_output,
                        samples[j])
                proportions_grad[i] += 

    def _proportions_to_weights_grad(self, weights, proportions_grad):
        # TODO: Is our method of using the chain rule valid?
        weights_grad = np.empty(weights.shape)
        weights_sum = np.sum(weights)
        derivatives = -weights / np.sum(np.square(weights_sum))
        for i in xrange(weights.shape[0]):
            temp = derivatives.copy()
            temp[i] += weights_sum / np.square(weights_sum)
            weights_grad[i] = proportions_grad.dot(temp)

        return weights_grad

    def _gaussian_density(self, x, means, covariance):
        dimension = means.shape[0]
        return (np.power(2.0 * np.pi, -dimension / 2.0) *
                np.power(np.linalg.det(covariance), -0.5) * 
                np.exp(-0.5 * (((x - means).dot(np.linalg.inv(covariance))).
                dot(x - means))))

    def _generate_covariance_matrix(self, points, kernel):
        num_points = points.shape[0]
        covariance_matrix = np.empty([num_points, num_points])
        for i in xrange(num_points):
            for j in xrange(num_points):
                covariance_matrix[i, j] = kernel(points[i], points[j])

        return covariance_matrix
