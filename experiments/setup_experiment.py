"""A set of functions to setup and run individual experiments."""
import numpy as np

import data_source
import data_transformation
from savigp.kernel import ExtRBF
from savigp import likelihood
import run_model


def boston_experiment(method, components, sparsity_factor, run_id, optimize_stochastic=False):
    """
    Run the boston housing experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'boston'
    data = data_source.boston_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 1, True)
    cond_ll = likelihood.UnivariateGaussian(np.array(1.0))
    transform = data_transformation.MeanTransformation(data['train_inputs'], data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               True,
                               False,
                               optimization_config={'mog': 25, 'hyp': 25, 'll': 25, 'inducing': 8},
                               max_iter=200,
                               optimize_stochastic=optimize_stochastic)


def wisconsin_experiment(method, components, sparsity_factor, run_id, optimize_stochastic=False):
    """
    Run the wisconsin experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'breast_cancer'
    data = data_source.wisconsin_breast_cancer_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 1, False)
    cond_ll = likelihood.LogisticLL()
    transform = data_transformation.IdentityTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               True,
                               False,
                               optimization_config={'mog': 25, 'hyp': 25},
                               max_iter=200)


def mining_experiment(method, components, sparsity_factor, run_id, optimize_stochastic=False):
    """
    Run the mining experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'mining'
    data = data_source.mining_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 1, False)
    cond_ll = likelihood.LogGaussianCox(np.log(191. / 811))
    transform = data_transformation.IdentityTransformation(data['train_inputs'],
                                                           data['train_outputs'])
    kernel[0].variance = 1.0
    kernel[0].lengthscale = 13516.0

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               True,
                               True,
                               optimization_config={'mog': 15000},
                               max_iter=1)


def usps_experiment(method, components, sparsity_factor, run_id, optimize_stochastic=False):
    """
    Run the usps experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'usps'
    data = data_source.usps_data()[run_id - 1]
    kernel = [ExtRBF(data['train_inputs'].shape[1], variance=2, lengthscale=np.array((4.,)), ARD=False)
              for _ in range(3)]
    cond_ll = likelihood.SoftmaxLL(3)
    transform = data_transformation.IdentityTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               True,
                               False,
                               optimization_config={'mog': 25, 'hyp': 25},
                               max_iter=300)


def abalone_experiment(method, components, sparsity_factor, run_id, optimize_stochastic=False):
    """
    Run the abalone experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'abalone'
    data = data_source.abalone_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 1, False)
    cond_ll = likelihood.WarpLL(np.array([-2.0485, 1.7991, 1.5814]),
                                np.array([2.7421, 0.9426, 1.7804]),
                                np.array([0.1856, 0.7024, -0.7421]),
                                np.log(0.1))
    transform = data_transformation.MinTransformation(data['train_inputs'], data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               True,
                               False,
                               optimization_config={'mog': 25, 'hyp': 25, 'll': 25},
                               max_iter=200)


def creep_experiment(method, components, sparsity_factor, run_id, optimize_stochastic=False):
    """
    Run the creep experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'creep'
    data = data_source.creep_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 1, True)
    cond_ll = likelihood.WarpLL(np.array([3.8715, 3.8898, 2.8759]),
                                np.array([1.5925, -1.3360, -2.0289]),
                                np.array([0.7940, -4.1855, -3.0289]),
                                np.log(0.01))

    scaler = data_transformation.preprocessing.StandardScaler().fit(data['train_inputs'])
    data['train_inputs'] = scaler.transform(data['train_inputs'])
    data['test_inputs'] = scaler.transform(data['test_inputs'])
    transform = data_transformation.MinTransformation(data['train_inputs'], data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               True,
                               False,
                               optimization_config={'mog': 25, 'hyp': 25, 'll': 25},
                               max_iter=200)


def mnist_experiment(method, components, sparsity_factor, run_id,
                     image=None, n_threads=1, partition_size=3000,
                     optimize_stochastic=False):
    """
    Run the mnist experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'mnist'
    data = data_source.mnist_data()[run_id - 1]
    kernel = [ExtRBF(data['train_inputs'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False)
              for _ in range(10)]
    cond_ll = likelihood.SoftmaxLL(10)
    transform = data_transformation.IdentityTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 60, 'hyp': 15},
                               max_iter=300,
                               n_threads=n_threads,
                               ftol=10,
                               model_image_dir=image,
                               partition_size=partition_size,
                               optimize_stochastic=optimize_stochastic)


def mnist8m_experiment(method, components, sparsity_factor, run_id,
                       image=None, n_threads=1, partition_size=3000,
                       optimize_stochastic=True, num_samples=1000,
                       max_iter=8000):
    """
    Run the mnist8m experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'mnist8m'
    data = data_source.mnist8m_data()[run_id - 1]
    kernel = [ExtRBF(data['train_inputs'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False)
              for _ in range(10)]
    cond_ll = likelihood.SoftmaxLL(10)
    transform = data_transformation.IdentityTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 60, 'hyp': 15, 'inducing': 6},
                               num_samples=num_samples,
                               max_iter=max_iter,
                               n_threads=n_threads,
                               ftol=10,
                               model_image_dir=image,
                               partition_size=partition_size,
                               optimize_stochastic=optimize_stochastic)


def mnist_binary_experiment(method, components, sparsity_factor, run_id,
                            image=None, n_threads=1, partition_size=3000,
                            optimize_stochastic=False):
    """
    Run the binary mnist experiment.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'mnist_binary'
    data = data_source.mnist_binary_data()[run_id - 1]
    kernel = [ExtRBF(data['train_inputs'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False)]
    cond_ll = likelihood.LogisticLL()
    transform = data_transformation.IdentityTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 12, 'hyp': 1, 'inducing': 1},
                               max_iter=300,
                               n_threads=n_threads,
                               ftol=10,
                               model_image_dir=image,
                               partition_size=partition_size,
                               optimize_stochastic=optimize_stochastic)


def mnist_binary_inducing_experiment(method, components, sparsity_factor, run_id,
                                     image=None, n_threads=1, partition_size=3000,
                                     optimize_stochastic=False):
    """
    Run the binary mnist experiment with inducing point learning.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'mnist_binary'
    data = data_source.mnist_binary_data()[run_id - 1]
    kernel = [ExtRBF(data['train_inputs'].shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False)]
    cond_ll = likelihood.LogisticLL()
    transform = data_transformation.IdentityTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 60, 'hyp': 15, 'inducing': 6},
                               max_iter=9,
                               n_threads=n_threads,
                               ftol=10,
                               model_image_dir=image,
                               partition_size=partition_size,
                               optimize_stochastic=optimize_stochastic)

def sarcos_experiment(method, components, sparsity_factor, run_id,
                      image=None, n_threads=1, partition_size=3000,
                      optimize_stochastic=False):
    """
    Run the sarcos experiment on two joints.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'sarcos'
    data = data_source.sarcos_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 3, False)
    cond_ll = likelihood.CogLL(0.1, 2, 1)
    transform = data_transformation.MeanStdYTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 25, 'hyp': 10, 'll': 10},
                               max_iter=200,
                               partition_size=partition_size,
                               n_threads=n_threads,
                               model_image_dir=image,
                               optimize_stochastic=optimize_stochastic)


def sarcos_inducing_experiment(method, components, sparsity_factor, run_id,
                              image=None, n_threads=1, partition_size=3000,
                              optimize_stochastic=False):
    """
    Run the sarcos experiment on two joints.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'sarcos'
    data = data_source.sarcos_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 3, False)
    cond_ll = likelihood.CogLL(0.1, 2, 1)
    transform = data_transformation.MeanStdYTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               True,
                               False,
                               optimization_config={'mog': 5, 'hyp': 2, 'll': 2, 'inducing': 1},
                               max_iter=200,
                               partition_size=partition_size,
                               n_threads=n_threads,
                               model_image_dir=image,
                               optimize_stochastic=optimize_stochastic)


def sarcos_all_joints_experiment(method, components, sparsity_factor, run_id,
                                 image=None, n_threads=1, partition_size=3000,
                                 optimize_stochastic=False):
    """
    Run the sarcos experiment on all joints.

    Parameters
    ----------
    method : str
        The method under which to run the experiment (mix1, mix2, or full).
    sparsity_factor : float
        The sparsity of inducing points.
    run_id : int
        The id of the configuration.
    """
    name = 'sarcos_all_joints'
    data = data_source.sarcos_all_joints_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 8, False)
    cond_ll = likelihood.CogLL(0.1, 7, 1)

    scaler = data_transformation.preprocessing.StandardScaler().fit(data['train_inputs'])
    data['train_inputs'] = scaler.transform(data['train_inputs'])
    data['test_inputs'] = scaler.transform(data['test_inputs'])
    transform = data_transformation.MeanStdYTransformation(data['train_inputs'],
                                                           data['train_outputs'])

    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 25, 'hyp': 10, 'll': 10, 'inducing': 6},
                               max_iter=200,
                               partition_size=partition_size,
                               ftol=10,
                               n_threads=n_threads,
                               model_image_dir=image,
                               optimize_stochastic=optimize_stochastic)


def airline_experiment(method, components, sparsity_factor, run_id,
                       image=None, n_threads=1, partition_size=3000,
                       optimize_stochastic=False):
    name = 'airline'
    data = data_source.airline_data()[run_id - 1]
    kernel = get_kernels(data['train_inputs'].shape[1], 1, True)
    cond_ll = likelihood.UnivariateGaussian(np.array(1.0))

    transform = data_transformation.MeanStdTransformation(data['train_inputs'],
                                                          data['train_outputs'])
    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 5, 'hyp': 2, 'll': 1, 'inducing': 1},
                               max_iter=200,
                               partition_size=partition_size,
                               ftol=10,
                               n_threads=n_threads,
                               model_image_dir=image,
                               optimize_stochastic=optimize_stochastic)


def get_kernels(input_dim, num_latent_proc, ARD):
    return [ExtRBF(input_dim, variance=1, lengthscale=np.array((1.,)), ARD=ARD)
            for _ in range(num_latent_proc)]


def seismic_experiment(method, components, sparsity_factor, run_id,
                       image=None, n_threads=1, partition_size=3000,
                       optimize_stochastic=False):
    name = 'seismic'
    data = data_source.seismic_data()[0]
    kernel = get_kernels(data['train_inputs'].shape[1], 8, True)
    cond_ll = likelihood.SeismicLL(4)

    transform = data_transformation.MeanStdTransformation(data['train_inputs'],
                                                          data['train_outputs'])
    return run_model.run_model(data['train_inputs'],
                               data['train_outputs'],
                               data['test_inputs'],
                               data['test_outputs'],
                               cond_ll,
                               kernel,
                               method,
                               components,
                               name,
                               data['id'],
                               sparsity_factor,
                               transform,
                               False,
                               False,
                               optimization_config={'mog': 5, 'hyp': 2, 'inducing': 1},
                               max_iter=200,
                               partition_size=partition_size,
                               ftol=10,
                               n_threads=n_threads,
                               model_image_dir=image,
                               optimize_stochastic=optimize_stochastic)

if __name__ == '__main__':
    seismic_experiment('full', 1, 0.1, 1, None)
