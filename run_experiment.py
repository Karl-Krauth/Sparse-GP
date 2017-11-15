#!/usr/bin/env python
"""A script to run a set of gaussian process experiments under differing configurations."""

import argparse
import json
from multiprocessing.pool import Pool

import numpy as np

from experiments import setup_experiment

EXPERIMENTS = {
    'airline': setup_experiment.airline_experiment,
    'boston': setup_experiment.boston_experiment,
    'wisconsin': setup_experiment.wisconsin_experiment,
    'mining': setup_experiment.mining_experiment,
    'usps': setup_experiment.usps_experiment,
    'abalone': setup_experiment.abalone_experiment,
    'creep': setup_experiment.creep_experiment,
    'mnist': setup_experiment.mnist_experiment,
    'mnist8m': setup_experiment.mnist8m_experiment,
    'mnist_binary': setup_experiment.mnist_binary_experiment,
    'mnist_binary_inducing': setup_experiment.mnist_binary_inducing_experiment,
    'sarcos': setup_experiment.sarcos_experiment,
    'sarcos_inducing': setup_experiment.sarcos_inducing_experiment,
    'sarcos_all_joints': setup_experiment.sarcos_all_joints_experiment,
    'seismic': setup_experiment.seismic_experiment,
}

METHODS = ['diag', 'full']


def main():
    """Run the experiments requested by the user."""
    args = setup_args()
    np.random.seed(1)

    if 'file' in args:
        # Run multiple experiments from a configuration file.
        with open(args['file']) as config_file:
            config = json.loads(config_file.read())
            run_parallel(**config)
    elif 'experiment_name' in args:
        # Run a single experiment from command line arguments.
        experiment = EXPERIMENTS[args['experiment_name']]
        del args['experiment_name']
        if args['method'] == 'full' and args['components'] != 1:
            print 'Only one components allowed for full Gaussian posterior.'
        else:
            experiment(**args)
    else:
        print 'You must either chose an experiment (-e) or a config file (-f).'


def setup_args():
    """Get the commandline arguments and return them in a dictionary."""
    parser = argparse.ArgumentParser(description='Experimental framework for savigp.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--experiment_name', choices=EXPERIMENTS, default=argparse.SUPPRESS,
                        help='The name of the experiment to run.')
    parser.add_argument('-f', '--file', default=argparse.SUPPRESS,
                        help='A json file containing a list of experiment configurations to run.')
    parser.add_argument('-m', '--method', choices=METHODS, default=METHODS[0],
                        help='The type of mixture of gaussians to learn.')
    parser.add_argument('-c', '--components', type=int, default=1,
                        help='The number of components to use in the mixture of Gaussians.')
    parser.add_argument('-s', '--sparsity_factor', type=float, default=1.0,
                        help='The sparsity of inducing points. Value must be between 0 and 1.')
    parser.add_argument('-r', '--run_id', type=int, default=1,
                        help='The id of the experiment configuration.')
    parser.add_argument('-i', '--image', default=argparse.SUPPRESS,
                        help='A path to a partially completed large scale experiment')
    parser.add_argument('-n', '--n_threads', type=int, default=argparse.SUPPRESS,
                        help='The number of threads to run for a large scale experiment.')
    parser.add_argument('-p', '--partition_size', type=int, default=argparse.SUPPRESS,
                        help='The size of sample partitions for a large scale experiment.')
    parser.add_argument('-o', '--optimize_stochastic', action='store_true',
                        help='Whether to optimize the model stochastically.')
    return vars(parser.parse_args())


def run_parallel(num_processes, experiment_names, methods, sparsity_factors, run_ids):
    """
    Run multiple experiments in parallel.

    Parameters
    ----------
    num_processes : int
        The maximum number of processes that can run concurrently.
    experiment_names : list of str
        The names of experiments to run.
    methods : list of str
        The methods to run the experiments under (mix1, mix2, or full).
    sparsity_factors : list of float
        The sparsity of inducing points to run the experiments at.
    run_ids : list of int
        The ids of the configurations under which to run the experiments.
    """
    # Setup an array of individual experiment configurations.
    experiment_configs = []
    for experiment in experiment_names:
        for method in methods:
            for sparsity_factor in sparsity_factors:
                for run_id in run_ids:
                    experiment_configs.append({'experiment_name': experiment,
                                               'method': method,
                                               'sparsity_factor': sparsity_factor,
                                               'run_id': run_id})

    # Now run the experiments.
    pool = Pool(num_processes)
    pool.map(run_config, experiment_configs)


def run_config(config):
    """Runs an experiment under the given configuration."""
    experiment = EXPERIMENTS[config['experiment_name']]
    del config['experiment_name']
    experiment(**config)


if __name__ == '__main__':
    main()

