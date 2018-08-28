"""This module contain utilities for optimizing the parameters of a gaussian process model."""

import math
import time

import numpy as np
import scipy.optimize

import gaussian_process
import model_logging
import util

# The parameter sets that must be bounded during optimization.
PARAMETER_SETS_TO_BOUND = ['hyp', 'inducing']


def batch_optimize_model(model, optimization_config, max_iterations=200, mog_threshold=1e-4,
                         objective_threshold=1e-5):
    """
    Optimise the model by using batch gradient descent alternating the optimization through
    sets of parameters.

    Parameters
    ----------
    model : model
        The model to optimise.
    optimization_config : dict
        The maximum number of function evaluations per subset of parameters for each local
        optimization. Valid keys are contained in PARAMETER_SETS. If a set of
        parameters is not included it will not get optimized.
    max_iterations : int
        The maximum number of global optimisations. If max_iterations is None the model will keep
        getting optimized until convergence.
    mog_threshold : float
        The tolerance threshold for convergence of the mixture of gaussian parameters. The model
        is considered to have converged if (means_difference + covariances_difference) / 2 <
        mog_tolerance.
    objective_threshold : float
        The tolerance threshold for convergence of the objective function.
    """
    start = time.time()
    current_iter = 1
    total_evaluations = 0
    checker = ConvergenceChecker(mog_threshold, objective_threshold)

    try:
        while (not checker.is_converged() and
                (max_iterations is None or current_iter <= max_iterations)):
            model_logging.logger.info('Iteration %d started.', current_iter)

            # Go through and optimize each set of parameters in optimization_config.
            for parameter_set in gaussian_process.PARAMETER_SETS:
                if parameter_set in optimization_config:
                    model_logging.logger.info('Optimizing: %s', parameter_set)
                    model.set_optimization_method(parameter_set)
                    total_evaluations += bfgs(model, max_fun=optimization_config[parameter_set],
                                              apply_bounds=parameter_set in PARAMETER_SETS_TO_BOUND)

            # Update and save the state of the optimization.
            model_logging.snapshot_model(model)
            checker.update_parameters(model.objective_function(),
                                      *model.get_gaussian_mixture_params())
            current_iter += 1
    except KeyboardInterrupt:
        model_logging.logger.info('Interrupted by the user.')
        model_logging.logger.info('Last objective value: %f', model.objective_function())

    end = time.time()

    return (end - start), total_evaluations


def bfgs(model, max_fun=None, apply_bounds=False):
    """
    Optimise the model using the l_bfgs_b algorithm.

    Parameters
    ----------
    model : model
        The model to optimise.
    max_fun : int (optional)
        The maximum number of evaluations.
    apply_bounds : boolean (optional)
        Whether to apply bounds. If True, parameters will be limited to be less than log(1e10).
    """
    wrapper = BatchModelWrapper(model)
    starting_params = model.get_params()
    bounds = None
    if apply_bounds:
        bounds = [(None, math.log(1e10)) for _ in xrange(starting_params.shape[0])]

    for num_tries in xrange(4):
        try:
            scipy.optimize.fmin_l_bfgs_b(
                func=wrapper.objective_function,
                x0=starting_params,
                fprime=wrapper.objective_function_gradients,
                factr=5,
                maxfun=max_fun,
                callback=wrapper.update,
                bounds=bounds)
            wrapper.finish_optimization()
            break
        except OptTermination:
            model_logging.logger.warning('Invalid value encountered. Optimization restarted.')
            wrapper.reset()
            max_fun = 3 - num_tries

    return wrapper.get_total_function_evals()


class BatchModelWrapper(object):
    def __init__(self, model):
        self._model = model
        self._initial_params = model.get_params()
        self._last_params = self._initial_params
        self._best_params = self._initial_params
        self._best_target_val = self._model.objective_function()
        self._total_function_evals = 0

    def update(self, new_params):
        if np.array_equal(new_params, self._last_params):
            return

        try:
            self._model.set_params(new_params.copy())
        except (ValueError, util.JitChol) as exception:
            raise OptTermination(exception)

        self._total_function_evals += 1
        self._last_params = new_params.copy()

        if self._model.objective_function() < self._best_target_val:
            self._best_target_val = self._model.objective_function()
            self._best_params = new_params.copy()

    def objective_function(self, new_params):
        self.update(new_params)
        model_logging.logger.debug('Objective: %.4f - KL: %.4f - ELL: %.4f', self._model.objective_function(),
                                   -(self._model.cached_entropy + self._model.cached_cross),
                                     self._model.cached_ell)
        return self._model.objective_function().astype(np.float64)

    def objective_function_gradients(self, new_params):
        self.update(new_params)
        return self._model.objective_function_gradients().astype(np.float64)

    def reset(self):
        self.update(self._initial_params)
        self.__init__(self._model)

    def finish_optimization(self):
        return self.update(self._best_params)

    def get_total_function_evals(self):
        return self._total_function_evals


def stochastic_optimize_model(model, optimization_config, max_iterations=200, mog_threshold=1e-4,
                              objective_threshold=1e-5, num_batches=1):
    """
    Optimise the model using mini-batch stochastic gradient descent by alternating the optimization
    through sets of parameters.

    Parameters
    ----------
    model : model
        The model to optimise.
    optimization_config : dict
        The number of passes through the data per subset of parameters for each local
        optimization. Valid keys are contained in gaussian_process.PARAMETER_SETS. If a set of
        parameters is not included it will not get optimized.
    max_iterations : int
        The maximum number of global optimisations. If max_iterations is None the model will keep
        getting optimized until convergence.
    mog_threshold : float
        The tolerance threshold for convergence of the mixture of gaussian parameters. The model
        is considered to have converged if (means_difference + covariances_difference) / 2 <
        mog_tolerance.
    objective_threshold : float
        The tolerance threshold for convergence of the objective function.
    num_batches : int
        The number of batches to consider during each gradient update. The batch size is
        defined by the model.
    """
    start = time.time()
    current_iter = 1
    total_evaluations = 0
    checker = ConvergenceChecker(mog_threshold, objective_threshold)

    try:
        while (not checker.is_converged() and
               (max_iterations is None or current_iter < max_iterations)):
            model_logging.logger.info('Iteration %d started.', current_iter)
            model.shuffle_data()

            # Go through and optimize each set of parameters in optimization_config.
            i = 0
            for parameter_set in gaussian_process.PARAMETER_SETS:
                if parameter_set in optimization_config:
                    model_logging.logger.info('Optimizing: %s', parameter_set)
                    model.set_optimization_method(parameter_set)
                    total_evaluations += sgd(model, num_batches, optimization_config[parameter_set], i)
                    i += 1

            # Update and save the state of the optimization.
            model_logging.snapshot_model(model)
            checker.update_parameters(model.overall_objective_function(),
                                      *model.get_gaussian_mixture_params())
            current_iter += 1
    except KeyboardInterrupt:
        model_logging.logger.info('Interrupted by the user.')
        model_logging.logger.info('Last objective value: %f', model.overall_objective_function())

    end = time.time()

    return (end - start), total_evaluations


grad_rms = [0] * 10
change_rms = [0] * 10
def sgd(model, num_batches, max_passes, idx):
    """
    Optimise the model using mini-batch stochastic gradient descent.

    Parameters
    ----------
    model : model
        The model to optimise.
    num_batches : int
        The number of batches to consider in one gradient update.
    max_passes : int
        The number of passes through the data.
    """
    num_evals = 0
    curr_train_index = 0
    global grad_rms
    global change_rms
    global iternum
    eps = 1e-6
    decay_rate = 0.95

    model.set_train_partitions(curr_train_index, num_batches)
    model.set_params(model.get_params())

    for i in xrange(max_passes):
        model.shuffle_data()
        for j in xrange(model.get_num_partitions() / num_batches):
            old_params = model.get_params()
            grad_rms[idx] = (decay_rate * grad_rms[idx] + (1 - decay_rate) *
                             model.objective_function_gradients() ** 2)
            change = -(np.sqrt(change_rms[idx] + eps) / np.sqrt(grad_rms[idx] + eps) *
                       model.objective_function_gradients())
            change_rms[idx] = decay_rate * change_rms[idx] + (1 - decay_rate) * change ** 2
            new_params = old_params + change

            curr_train_index += num_batches
            if curr_train_index == model.get_num_partitions():
                curr_train_index = 0
            model.set_train_partitions(curr_train_index, num_batches)
            model.set_params(new_params)

            model_logging.logger.debug('Objective: %.4f', model.objective_function())
            num_evals += 1

    return num_evals


class ConvergenceChecker(object):
    """
    A class to hold state relevant to convergence when optimizing a model.

    Parameters
    ----------
    mog_threshold : float
        The threshold that determines if the mixture of gaussian parameters have converged.
    objective_threshold : float
        The threshold that determines if the objective function has converged.
    """
    def __init__(self, mog_threshold, objective_threshold):
        self._mog_threshold = mog_threshold
        self._objective_threshold = objective_threshold
        self._prev_means = None
        self._curr_means = None
        self._prev_covariances = None
        self._curr_covariances = None
        self._prev_objective_val = None
        self._curr_objective_val = None

    def update_parameters(self, new_objective_val, new_means, new_covariances):
        """
        Update the model parameters relevant to convergence.

        Parameters
        ----------
        new_objective_val : float
            The value of the objective function given the parameters in the current iteration.
        new_means : ndarray
            The value of the mixture of gaussian means in the current iteration.
        new_covariances : ndarray
            The value of the mixture of gaussian covariances in the current iteration.
        """
        # Update objective values.
        self._prev_objective_val = self._curr_objective_val
        self._curr_objective_val = new_objective_val
        # Update mixture of gaussian parameters.
        self._prev_means = self._curr_means
        self._curr_means = new_means
        self._prev_covariances = self._curr_covariances
        self._curr_covariances = new_covariances

    def is_converged(self):
        """
        Check if the model has converged.

        Returns
        -------
        bool
            True if the model has converged and False otherwise.
        """
        # Check if we have gone through less than two iterations.
        if self._prev_objective_val is None:
            return False

        # Calculate differences from the previous and the current iteration.
        objective_val_difference = self._prev_objective_val - self._curr_objective_val
        means_difference = np.absolute(self._curr_means - self._prev_means).mean()
        covariances_difference = np.absolute(self._curr_covariances - self._prev_covariances).mean()

        # Log the current differences.
        model_logging.logger.debug('Objective difference: %f', objective_val_difference)
        model_logging.logger.info('Mean difference: %f', means_difference)
        model_logging.logger.info('Covariance difference: %f', covariances_difference)

        # Check for convergence.
        if (self._prev_objective_val > self._curr_objective_val and
                objective_val_difference < self._objective_threshold):
            return True
        elif (means_difference + covariances_difference) / 2.0 < self._mog_threshold:
            return True
        else:
            return False


class OptTermination(Exception):
    """
    An exception to indicate problems during an optimisation. For example, problems in
    function evaluation.
    """
    pass
