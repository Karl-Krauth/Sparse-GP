"""
Provides facility for fitting models to data, making predictions and exporting results to csv files.
"""
import os
import pickle

import model_logging
import optimizer
from diagonal_gaussian_process import DiagonalGaussianProcess
from full_gaussian_process import FullGaussianProcess
import util

def run_model(train_X,
              train_Y,
              test_X,
              test_Y,
              cond_ll,
              kernel,
              method,
              name,
              run_id,
              sparsity_factor,
              trans_class,
              random_Z,
              export_X,
              optimization_config={'mog': 25, 'hyp': 25, 'll': 25},
              num_samples=2000,
              latent_noise=0.001,
              max_iter=200,
              n_threads=1,
              model_image_dir=None,
              xtol=1e-3,
              ftol=1e-5,
              partition_size=3000):
    """
    Fit a model to the data (train_X, train_Y) using the method provided by 'method', and make
    predictions on 'test_X' and 'test_Y', and export the result to csv files.

    Parameters
    ----------
    train_X : ndarray
        X of training points
    train_Y : ndarray
        Y of traiing points
    test_X : ndarray
        X of test points
    test_Y : ndarray
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
    transformer = trans_class.get_transformation(train_Y, train_X)
    train_Y = transformer.transform_Y(train_Y)
    test_Y = transformer.transform_Y(test_Y)
    train_X = transformer.transform_X(train_X)
    test_X = transformer.transform_X(test_X)

    num_inducing = int(train_X.shape[0] * sparsity_factor)
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

    model_image = None
    if model_image_dir is not None:
        model_image_file_path = os.path.join(model_image_dir, 'model.dump')
        with open(model_image_file_path) as model_image_file:
            model_image = pickle.load(model_image_file)

    if model_image:
        model_logging.logger.info('loaded model - iteration started from: ' +
                                  str(opt_params['current_iter']))

    if method == 'full':
        model = FullGaussianProcess(train_X, train_Y, num_inducing, num_samples, kernel, cond_ll,
                                    latent_noise, False, random_Z,
                                    num_threads=n_threads)
    elif method == 'mix1' or method == 'mix2':
        num_components = 1 if method == 'mix1' else 2
        model = DiagonalGaussianProcess(train_X, train_Y, num_inducing, num_components, num_samples,
                                        kernel, cond_ll, latent_noise, False, random_Z,
                                        num_threads=n_threads)

    properties['total_time'], properties['total_evals'] = optimizer.optimize_model(model, optimization_config, max_iter,
                                                                                   xtol, ftol)

    model_logging.logger.debug("prediction started...")
    y_pred, var_pred, nlpd = model.predict(test_X, test_Y)
    model_logging.logger.debug("prediction finished")

    model_logging.export_training_data(transformer.untransform_X(train_X),
                                       transformer.untransform_Y(train_Y),
                                       export_X)
    model_logging.export_predictions(transformer.untransform_X(test_X),
                                     transformer.untransform_Y(test_Y),
                                     [transformer.untransform_Y(y_pred)],
                                     [transformer.untransform_Y_var(var_pred)],
                                     transformer.untransform_NLPD(nlpd),
                                     export_X)

    model_logging.export_configuration(properties)

    return model
