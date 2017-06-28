"""
Provides facility for fitting models to data, making predictions and exporting results to csv files.
"""
import cPickle
import os

import numpy as np

from savigp import model_logging
from savigp import optimizer
from savigp.diagonal_gaussian_process import DiagonalGaussianProcess
from savigp.full_gaussian_process import FullGaussianProcess
from savigp import util

def run_model(train_inputs,
              train_outputs,
              test_inputs,
              test_outputs,
              cond_ll,
              kernel,
              method,
              num_components,
              name,
              run_id,
              sparsity_factor,
              transformer,
              random_Z,
              export_X,
              optimization_config,
              num_samples=2000,
              latent_noise=0.001,
              max_iter=200,
              n_threads=1,
              model_image_dir=None,
              xtol=1e-3,
              ftol=1e-5,
              partition_size=3000,
              optimize_stochastic=False):
    """
    Fit a model to the data (train_X, train_Y) using the method provided by 'method', and make
    predictions on 'test_X' and 'test_Y', and export the result to csv files.

    Parameters
    ----------
    train_inputs : ndarray
        X of training points
    train_outputs : ndarray
        Y of training points
    test_inputs : ndarray
        X of test points
    test_outputs : ndarray
        Y of test points
    cond_ll : subclass of likelihood/Likelihood
        Conditional log likelihood function used to build the model.
    kernel : list
        The kernel that the model uses. It should be an array, and size of the array should be same
        as the number of latent processes. Each element should provide interface similar to ``ExtRBF``
        class.
    method : string
        The method to use to learns the model. It can be 'full', 'mix1', and 'mix2'
    name : string
        The name that will be used for logger file names, and results files names
    run_id : object
        ID of the experiment, which can be anything, and it will be included in the configuration
        file. It can provide for example a number referring to a particular test and train partition.
    num_samples : integer
        Number of samples for estimating objective function and gradients
    sparsity_factor : float
        Can be any number and will be included in the configuration file. It will not determine
        the number of inducing points
    trans_class : subclass of DataTransformation
        The class which will be used to transform data.
    random_Z : boolean
        Whether to initialise inducing points randomly on the training data. If False, inducing points
        will be placed using k-means (or mini-batch k-mean) clustering. If True, inducing points will
        be placed randomly on the training data.
    export_X : boolean
        Whether to export X to csv files.
    latent_noise : integer
        The amount of latent noise to add to the kernel. A white noise of amount latent_noise will be
        added to the kernel.
    opt_per_iter: integer
        Number of update of each subset of parameters in each iteration, e.g., {'mog': 15000,
        'hyp': 25, 'll': 25}
    max_iter: integer
        Maximum of global iterations used on optimization.
    n_threads: integer
        Maximum number of threads used.
    model_image_file: string
        The image file from the which the model will be initialized.
    xtol: float
        Tolerance of 'X' below which the optimization is determined as converged.
    ftol: float
        Tolerance of 'f' below which the optimization is determined as converged.
    partition_size: integer
        The size which is used to partition training data (This is not the partition used for SGD).
        Training data will be split to the partitions of size ``partition_size`` and calculations will
        be done on each partition separately. This aim of this partitioning of data is to make
        algorithm memory efficient.

    Returns
    -------
    folder : string
        the name of the folder in which results are stored
    model : model
        the fitted model itself.
    """
    # Temporarily transform all training and test data.
    train_outputs = transformer.transform_Y(train_outputs)
    test_outputs = transformer.transform_Y(test_outputs)
    train_inputs = transformer.transform_X(train_inputs)
    test_inputs = transformer.transform_X(test_inputs)
    num_samples = 10000
    # Compute the number of inducing points from the sparsity factor.
    num_inducing = int(train_inputs.shape[0] * sparsity_factor)

    # Initialize and print experiment info.
    git_hash, git_branch = util.get_git()
    properties = {
        'method': method,
        'sparsity_factor': sparsity_factor,
        'sample_num': num_samples,
        'll': cond_ll.__class__.__name__,
        'optimization_config': optimization_config,
        'xtol': xtol,
        'ftol': ftol,
        'run_id': run_id,
        'experiment': name,
        'max_iter': max_iter,
        'git_hash': git_hash,
        'git_branch': git_branch,
        'random_Z': random_Z,
        'latent_noise:': latent_noise,
        'model_init': model_image_dir
    }
    model_logging.init_logger(name)
    model_logging.logger.info('experiment started for:' + str(properties))

    # Initialize the model.
    if model_image_dir is not None:
        model_image_file_path = os.path.join(model_image_dir, 'model.dump')
        with open(model_image_file_path) as model_image_file:
            model = cPickle.load(model_image_file)
    elif method == 'full':
        model = FullGaussianProcess(train_inputs,
                                    train_outputs,
                                    num_inducing,
                                    num_samples,
                                    kernel,
                                    cond_ll,
                                    latent_noise,
                                    False,
                                    random_Z,
                                    num_threads=n_threads,
                                    partition_size=partition_size)
    elif method == 'diag':
        model = DiagonalGaussianProcess(train_inputs,
                                        train_outputs,
                                        num_inducing,
                                        num_components,
                                        num_samples,
                                        kernel,
                                        cond_ll,
                                        latent_noise,
                                        False,
                                        random_Z,
                                        num_threads=n_threads,
                                        partition_size=partition_size)
    else:
        assert False

    def cb():
        pred_y, _, _ = model.predict(test_inputs, test_outputs)
        pred_y = transformer.untransform_Y(pred_y)
        t_out = transformer.untransform_Y(test_outputs)
        print "MSSE", ((pred_y - t_out) ** 2).mean() / ((t_out - t_out.mean(0)) ** 2).mean()
    optimizer.callback = cb 
    # Optimize the model.
    if optimize_stochastic:
        properties['total_time'], properties['total_evals'] = optimizer.stochastic_optimize_model(
            model, optimization_config, max_iter, xtol, ftol)
    else:
        properties['total_time'], properties['total_evals'] = optimizer.batch_optimize_model(
            model, optimization_config, max_iter, xtol, ftol)

    # Make predictions on test point.
    model_logging.logger.debug("prediction started...")
    y_pred, var_pred, nlpd = model.predict(test_inputs, test_outputs)
    model_logging.logger.debug("prediction finished")

    # Export untransformed result data.
    model_logging.export_training_data(transformer.untransform_X(train_inputs),
                                       transformer.untransform_Y(train_outputs),
                                       export_X)
    model_logging.export_predictions(transformer.untransform_X(test_inputs),
                                     transformer.untransform_Y(test_outputs),
                                     [transformer.untransform_Y(y_pred)],
                                     [transformer.untransform_Y_var(var_pred)],
                                     transformer.untransform_NLPD(nlpd),
                                     export_X)
    model_logging.export_configuration(properties)

    return model
