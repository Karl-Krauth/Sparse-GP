from __future__ import division

import numpy as np
import random
import scipy.linalg as la

np.random.seed(1)
random.seed(1)

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

        # Choose the inducing points by randomly selecting amongst samples.
        # Each latent function has num_inducing points.
        inducing_points = np.empty([self._num_latent, self._num_inducing,
                                    num_features])
        for i in xrange(self._num_latent):
            # TODO: This can be sped-up. Also do we really need different
            # inducing points per latent function?
            inducing_points[i] = np.random.permutation(
                training_inputs)[:self._num_inducing]

        # Initialize the mixture of gaussian parameters. We have a mixture
        # with num_components. Each component consists of num_latent gaussians
        # each of which is a num_inducing dimensional gaussian.
        # TODO: Maybe we should initialize things to other values?
        mog_means = np.ones([self._num_components, self._num_latent,
                              self._num_inducing])
        mog_covars = np.empty([self._num_components, self._num_latent,
                               self._num_inducing, self._num_inducing])
        mog_weights = np.ones([self._num_components])
        mog_proportions = np.exp(mog_weights) / np.sum(np.exp(mog_weights))
        for i in xrange(self._num_components):
            for j in xrange(self._num_latent):
                mog_covars[i, j] = np.eye(self._num_inducing)
        print inducing_points
        # Pre-compute the covariance matrix of the inducing points and its
        # inverse and determinant. Since the matrices are block diagonal we
        # store them as num_latent seperate sub-matrices of size num_inducing *
        # num_inducing.
        inducing_covars = np.empty([self._num_latent, self._num_inducing,
                                    self._num_inducing])
        inducing_covars_inverse = np.empty(inducing_covars.shape)
        det_inducing_covars = np.empty([self._num_latent])
        for i in xrange(self._num_latent):
            inducing_covars[i] = self._generate_covar_matrix(inducing_points[i],
                                                             self._kernels[i])
            # TODO: Can we take advantage of the symmetry of the matrix?
            # How do we deal with singular matrices? 
            inducing_covars_inverse[i] = np.linalg.inv(inducing_covars[i]) 
            det_inducing_covars[i] = np.linalg.det(inducing_covars[i])
        print "inducing covars:"
        print inducing_covars
        print "inducing covars inverse:"
        print inducing_covars_inverse
        inc = self._step_size + 1
        i = 0
        while inc > self._step_size:# and i < 2:
            i += 1
            index = random.randint(0, num_training - 1)
            inc = self._mog_gradient_descent(mog_means,
                                             mog_covars,
                                             mog_weights,
                                             mog_proportions,
                                             training_inputs[index],
                                             training_outputs[index],
                                             inducing_points,
                                             inducing_covars,
                                             inducing_covars_inverse,
                                             det_inducing_covars)
            print "means\n", mog_means
            print "covars\n", mog_covars
            print "weights\n", mog_weights
            print "props\n", mog_proportions
            print "inc\n", inc

    def predict(self, X):
        pass

    def _mog_gradient_descent(self, mog_means, mog_covars, mog_weights,
                              mog_proportions, training_input, training_output,
                              inducing_points, inducing_covars,
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
        means_grad += ent_means_grad
        covars_grad += ent_covars_grad
        weights_grad += ent_weights_grad

        ell_means_grad, ell_covars_grad, ell_weights_grad = (
            self._ell_grad(mog_means, mog_covars, mog_weights, mog_proportions,
                           training_input, training_output, inducing_points,
                           inducing_covars_inverse))
        means_grad += ell_means_grad
        covars_grad += ell_covars_grad
        weights_grad += ell_weights_grad

        # Update the parameters.
        mog_means += means_grad
        mog_covars += covars_grad
        mog_weights += weights_grad
        mog_proportions = np.exp(mog_weights) / np.sum(np.exp(mog_weights))
 
        return max(np.linalg.norm(means_grad), np.linalg.norm(covars_grad),
                   np.linalg.norm(weights_grad))

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
        print weights_grad
        print "props grad"
        print proportions_grad, "\n\n\n"

        return (means_grad, covars_grad, weights_grad)

    def _ell_grad(self, mog_means, mog_covars, mog_weights, mog_proportions,
                  train_input, train_output, inducing_points,
                  inducing_covars_inv):
        # Calculate intermediate values on the training point.
        train_covars = np.empty([self._num_latent, self._num_inducing])
        train_products = np.empty([self._num_latent, self._num_inducing])
        for i in xrange(self._num_latent):
            train_covars[i] = self._generate_covar_vector(train_input,
                                                          inducing_points[i],
                                                          self._kernels[i])
            train_products[i] = train_covars[i].dot(inducing_covars_inv[i])

        means_grad = np.zeros(mog_means.shape)
        covars_grad = np.zeros(mog_covars.shape)
        proportions_grad = np.zeros(mog_proportions.shape)
        for i in xrange(self._num_components):
            # Calculate the sample gaussian parameters.
            sample_means = np.empty([self._num_latent])
            sample_vars = np.empty([self._num_latent])
            for j in xrange(self._num_latent):
                sample_means[j] = train_products[j].dot(mog_means[i, j])
                sample_vars[j] = ((self._kernels[j](train_input,
                                 train_input) - train_products[j].
                                 dot(train_covars[j])) + train_products[j].
                                 dot(mog_covars[i, j]).dot(train_products[j]))

            # Generate the samples.
            samples = np.empty([self._num_gaussian_samples, self._num_latent])
            for j in xrange(self._num_gaussian_samples):
                for k in xrange(self._num_latent):
                    samples[j, k] = np.random.normal(sample_means[k],
                                                     sample_vars[k])

            # Now Calculate the gradients.
            for j in xrange(self._num_gaussian_samples):
                for k in xrange(self._num_latent):
                    means_grad[i, k] += (
                        (mog_proportions[i] / self._num_gaussian_samples) *
                        (inducing_covars_inv[k].dot(train_covars[k]) /
                        sample_vars[k]) * (samples[j, k] - sample_means[k]) *
                        self._likelihood(train_output, samples[j]))
                    covars_grad[i, k] += (
                        (mog_proportions[i] / (2.0 * self._num_gaussian_samples)) *
                        np.outer(train_products[k], train_products[k]) *
                        (np.power(sample_vars[k], -2) *
                        np.power((samples[j, k] - sample_means[k]), 2.0) -
                        1.0 / sample_vars[k]) * self._likelihood(train_output,
                        samples[j]))
                proportions_grad[i] += (self._likelihood(train_output, 
                                        samples[j]) / self._num_gaussian_samples)

        weights_grad = self._proportions_to_weights_grad(mog_weights,
                                                         proportions_grad)

        """
        print "============================="
        print "=============ELL============="
        print "============================="
        print "ent means grad:"
        print means_grad, "\n\n"
        print "covars grad"
        print covars_grad, "\n\n"
        print "weights grad"
        print weights_grad
        print "props grad"
        print proportions_grad, "\n\n\n"
        """
        return (means_grad, covars_grad, weights_grad)
        

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

    def _generate_covar_vector(self, point, points, kernel):
        num_points = points.shape[0]
        covar_vector = np.empty([num_points])
        for i in xrange(num_points):
            print point, points[i]
            covar_vector[i] = kernel(point, points[i])
        return covar_vector

    def _generate_covar_matrix(self, points, kernel):
        num_points = points.shape[0]
        covariance_matrix = np.empty([num_points, num_points])
        for i in xrange(num_points):
            for j in xrange(num_points):
                covariance_matrix[i, j] = kernel(points[i], points[j])

        return covariance_matrix
