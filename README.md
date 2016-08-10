# Sparse-GP #
An implementation of the model described by *TODO ADD JOURNAL REF* along with the framework to run
the experiments described in the paper.

## Table of Contents ##
[Dependencies](#dependencies)

[Running experiments](#running experiments)

[Experiment results](#experiment results)

[Using the model](#using the model)

[Architecture](#architecture)

## Dependencies ##
We require the following packages that can be directly installed from pip:
* GPy >= 1.0.7
* matplotlib >= 1.5.1
* numpy >= 1.11.1
* scikit-learn >= 0.17.1
* scipy >= 0.17.0
* pandas >= 0.18.0

We also require a Theano installation whose installation instructions can be found
[here](http://deeplearning.net/software/theano/install.html). We recommend configuring Theano to
[use GPUs](http://deeplearning.net/software/theano/tutorial/using_gpu.html) which will cause the
code to perform an order of magnitude faster.

## Running experiments ##
The script found at `src/run_experiments.py` allows us to run the experiments described in the
paper. `run_experiment.py` makes use of various flags to specify the configuration of the
experiment. For example
```
./src/run_experiment.py -e mnist -m full -s 0.04 -o -p 500
```
makes predictions on the mnist dataset, with a full covariance posterior, a sparsity factor
of 0.04, stochastic optimization with a minibatch of size 500. For full details on each flag run
```
./src/run_experiment.py -h
```
We also support launching multiple experiments at once, although we recommend only doing so for
small scale experiments. To run multiple experiments at once create a json file which contains the
following attributes:
* *num_processes*: The number of experiments to run at once.
* *experiment_names*: A list of names of the datasets to use.
* *methods*: A list of methods to use for the experiment.
* *run_ids*: The id of the dataset partition to use.

Then run
```
./src/run_experiment.py -f JSON_FILE_NAME
```
the script will then run all possible combinations of configurations found in the json file. An
example json file can be found in `./experiment_configs.json`.

## Experiment results ##
Each experiment that gets run generated a new directory in `../results` titled
`EXPERIMENT_NAME_DAY-MONTH-YEARhMINmSECs_PID`. The directory contains the following files:
* *config.csv*: The configuration information for the experiment.
* *EXPERIMENT_NAME.log*: Logging data during the optimization process.
* *model.dump*: A dump of the model that can be used to reload a regular snapshot of the model into
  memory.
* *predictions.csv*: The predictions the model made on the test data for the experiment.
* *train.csv*: A file containing training data used in the experiment.

## Using the model ##
The model can be used directly without the experiment framework. An example can be found in
`src/example.py`.

## Architecture ##
We give a quick summary of the design of the code. We can split the design into two sections:
the experiment framework, and the Gaussian process model.

### Experiment framework ###
The experiment framework provides the mean to load datasets into memory, optimizers, graph results,
and scripts to specify the configuration of experiments. The files in the experiment framework
consist of:
* *data_source.py*: Loads datasets held in `./data` into memory.
* *data_transformation.py*: Contains various utility functions to pre-process the data.
* *model_logging.py*: Logs experiment information found in `../results`.
* *optimizer.py*: Various functions to optimize the model.
* *run_experiment.py*: The script to allow users to run the experiment.
* *run_model.py*: Sets up the model, calls into the optimizer, makes predictions, and ensures
  all experiment info is logged to disk.
* *setup_experiment.py*: Configures experiments with information provided by `run_experiment.py`
  alongside extra details not provided by the user.

### Gaussian process model ###
The model consists of a generic Gaussian process superclass (`gaussian_process.py`), along with
two subclasses, one for a full covariance posterior (`full_gaussian_process.py`) and one for
a diagonal posterior (`diagonal_gaussian_process.py`).

The model also consists of classes to represent the posterior distribution (`gaussian_mixture.py`,
`full_gaussian_mixture.py`, and `diagonal_gaussian_mixture.py`), and a kernel function
implementation (`kernel.py`) and various likelihood models (`likelihood.py`).

We also hold various utility functions in `util.py`.
